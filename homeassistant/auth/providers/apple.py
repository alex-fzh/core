"""Sign in with Apple auth provider."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
import logging
from typing import Any

from aiohttp import ClientError
import jwt
import voluptuous as vol

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from ..models import AuthFlowContext, AuthFlowResult, Credentials, User, UserMeta
from . import AUTH_PROVIDER_SCHEMA, AUTH_PROVIDERS, AuthProvider, LoginFlow

_LOGGER = logging.getLogger(__name__)

APPLE_ISSUER = "https://appleid.apple.com"
APPLE_JWKS_URL = f"{APPLE_ISSUER}/auth/keys"
APPLE_JWKS_CACHE_TTL = timedelta(hours=24)
APPLE_ALGORITHM = "RS256"

CONF_ALLOWED_EMAILS = "allowed_emails"
CONF_ALLOWED_SUBJECTS = "allowed_subjects"
CONF_CLIENT_ID = "client_id"

CONFIG_SCHEMA = AUTH_PROVIDER_SCHEMA.extend(
    {
        vol.Required(CONF_CLIENT_ID): vol.Any(
            vol.All(str, vol.Length(min=1)),
            vol.All([vol.All(str, vol.Length(min=1))], vol.Length(min=1)),
        ),
        vol.Optional(CONF_ALLOWED_EMAILS): [vol.All(str, vol.Length(min=1))],
        vol.Optional(CONF_ALLOWED_SUBJECTS): [vol.All(str, vol.Length(min=1))],
    },
    extra=vol.PREVENT_EXTRA,
)


class InvalidAuthError(HomeAssistantError):
    """Raised when submitting invalid authentication."""


@AUTH_PROVIDERS.register("apple")
class AppleAuthProvider(AuthProvider):
    """Auth provider backed by Sign in with Apple identity tokens."""

    DEFAULT_TITLE = "Sign in with Apple"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Sign in with Apple auth provider."""
        super().__init__(*args, **kwargs)

        client_ids = self.config[CONF_CLIENT_ID]
        if isinstance(client_ids, str):
            self._client_ids = (client_ids,)
        else:
            self._client_ids = tuple(client_ids)

        self._allowed_emails = {
            email.casefold() for email in self.config.get(CONF_ALLOWED_EMAILS, [])
        }
        self._allowed_subjects = set(self.config.get(CONF_ALLOWED_SUBJECTS, []))
        self._jwks: dict[str, jwt.PyJWK] | None = None
        self._jwks_valid_until = dt_util.utcnow()

    async def async_login_flow(self, context: AuthFlowContext | None) -> AppleLoginFlow:
        """Return a flow to login."""
        return AppleLoginFlow(self)

    async def async_validate_identity_token(
        self, identity_token: str, name: str | None = None
    ) -> dict[str, str]:
        """Validate a Sign in with Apple identity token."""
        try:
            signing_key = await self._async_get_signing_key(identity_token)
            claims = jwt.decode(
                identity_token,
                key=signing_key,
                algorithms=[APPLE_ALGORITHM],
                audience=self._client_ids,
                issuer=APPLE_ISSUER,
                options={"require": ["aud", "exp", "iat", "iss", "sub"]},
            )
        except jwt.InvalidTokenError as err:
            raise InvalidAuthError("Invalid Apple identity token") from err

        subject = claims["sub"]
        if not isinstance(subject, str) or not subject:
            raise InvalidAuthError("Invalid Apple subject")

        if self._allowed_subjects and subject not in self._allowed_subjects:
            raise InvalidAuthError("Apple subject is not allowed")

        email = claims.get("email")
        if email is not None and not isinstance(email, str):
            raise InvalidAuthError("Invalid Apple email")

        email_verified = _claim_is_true(claims.get("email_verified"))
        if self._allowed_emails:
            if (
                email is None
                or not email_verified
                or email.casefold() not in self._allowed_emails
            ):
                raise InvalidAuthError("Apple email is not allowed")

        flow_result = {"subject": subject}
        if email:
            flow_result["email"] = email
            if email_verified:
                flow_result["email_verified"] = "true"
        if name:
            flow_result["name"] = name

        return flow_result

    async def _async_get_signing_key(self, identity_token: str) -> Any:
        """Return the signing key for an Apple identity token."""
        try:
            headers = jwt.get_unverified_header(identity_token)
        except jwt.InvalidTokenError as err:
            raise InvalidAuthError("Invalid Apple identity token header") from err

        if headers.get("alg") != APPLE_ALGORITHM:
            raise InvalidAuthError("Unexpected Apple identity token algorithm")

        key_id = headers.get("kid")
        if not isinstance(key_id, str) or not key_id:
            raise InvalidAuthError("Apple identity token key id is missing")

        jwks = await self._async_get_jwks()
        key = jwks.get(key_id)
        if key is None:
            jwks = await self._async_get_jwks(force_refresh=True)
            key = jwks.get(key_id)

        if key is None:
            raise InvalidAuthError("Apple identity token signing key is unknown")

        return key.key

    async def _async_get_jwks(
        self, *, force_refresh: bool = False
    ) -> dict[str, jwt.PyJWK]:
        """Fetch and cache Apple's JSON Web Key Set."""
        if (
            not force_refresh
            and self._jwks is not None
            and dt_util.utcnow() < self._jwks_valid_until
        ):
            return self._jwks

        session = async_get_clientsession(self.hass)

        try:
            async with session.get(APPLE_JWKS_URL) as response:
                response.raise_for_status()
                jwks = await response.json()
        except (ClientError, TimeoutError, ValueError) as err:
            _LOGGER.warning("Unable to fetch Sign in with Apple keys: %s", err)
            raise InvalidAuthError("Unable to fetch Apple signing keys") from err

        keys = jwks.get("keys")
        if not isinstance(keys, list):
            raise InvalidAuthError("Invalid Apple signing key response")

        parsed_keys: dict[str, jwt.PyJWK] = {}
        for key_data in keys:
            try:
                key = jwt.PyJWK.from_dict(key_data)
            except jwt.InvalidKeyError, TypeError, ValueError:
                continue

            key_id = key_data.get("kid")
            if isinstance(key_id, str):
                parsed_keys[key_id] = key

        if not parsed_keys:
            raise InvalidAuthError("Apple signing key response did not include keys")

        self._jwks = parsed_keys
        self._jwks_valid_until = dt_util.utcnow() + APPLE_JWKS_CACHE_TTL
        return parsed_keys

    async def async_get_or_create_credentials(
        self, flow_result: Mapping[str, str]
    ) -> Credentials:
        """Get credentials based on the flow result."""
        subject = flow_result["subject"]

        for credential in await self.async_credentials():
            if credential.data.get("subject") == subject:
                return credential

        data = {"subject": subject}
        for key in ("email", "name"):
            if key in flow_result:
                data[key] = flow_result[key]

        if flow_result.get("email_verified") == "true" and (
            user := await self._async_get_unique_homeassistant_user_for_email(
                flow_result.get("email")
            )
        ):
            credentials = self.async_create_credentials(data)
            await self.store.async_link_user(user, credentials)
            return credentials

        return self.async_create_credentials(data)

    async def _async_get_unique_homeassistant_user_for_email(
        self, email: str | None
    ) -> User | None:
        """Return one existing Home Assistant user with a matching email username."""
        if not email:
            return None

        normalized_email = email.casefold()
        matches = []

        for user in await self.store.async_get_users():
            if user.system_generated:
                continue

            for credential in user.credentials:
                if (
                    credential.auth_provider_type == "homeassistant"
                    and credential.data.get("username", "").casefold()
                    == normalized_email
                ):
                    matches.append(user)
                    break

        if len(matches) == 1:
            return matches[0]

        return None

    async def async_user_meta_for_credentials(
        self, credentials: Credentials
    ) -> UserMeta:
        """Return extra user metadata for credentials."""
        return UserMeta(
            name=credentials.data.get("name") or credentials.data.get("email"),
            is_active=True,
        )


class AppleLoginFlow(LoginFlow[AppleAuthProvider]):
    """Handler for the Sign in with Apple login flow."""

    async def async_step_init(
        self, user_input: dict[str, str] | None = None
    ) -> AuthFlowResult:
        """Handle the Apple identity token form."""
        errors = {}

        if user_input is not None:
            try:
                flow_result = await self._auth_provider.async_validate_identity_token(
                    user_input["identity_token"], user_input.get("name")
                )
            except InvalidAuthError:
                errors["base"] = "invalid_auth"
            else:
                return await self.async_finish(flow_result)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required("identity_token"): str,
                    vol.Optional("name"): str,
                }
            ),
            errors=errors,
        )


def _claim_is_true(value: Any) -> bool:
    """Return whether an Apple boolean-like claim is true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.casefold() == "true"
    return False
