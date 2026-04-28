"""Tests for the Sign in with Apple auth provider."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
import uuid

from cryptography.hazmat.primitives.asymmetric import rsa
import jwt
import pytest
import voluptuous as vol

from homeassistant import data_entry_flow
from homeassistant.auth import AuthManager, auth_store, models as auth_models
from homeassistant.auth.providers import apple, auth_provider_from_config
from homeassistant.core import HomeAssistant

CLIENT_ID = "com.example.VantageApp"
KEY_ID = "apple-test-key"


@pytest.fixture
async def store(hass: HomeAssistant) -> auth_store.AuthStore:
    """Mock store."""
    store = auth_store.AuthStore(hass)
    await store.async_load()
    return store


@pytest.fixture
def private_key() -> rsa.RSAPrivateKey:
    """Create a private key for signing test identity tokens."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def provider(
    hass: HomeAssistant, store: auth_store.AuthStore, private_key: rsa.RSAPrivateKey
) -> apple.AppleAuthProvider:
    """Mock provider."""
    provider = apple.AppleAuthProvider(
        hass,
        store,
        {
            "type": "apple",
            "client_id": CLIENT_ID,
        },
    )
    provider._async_get_signing_key = AsyncMock(return_value=private_key.public_key())
    return provider


@pytest.fixture
def manager(
    hass: HomeAssistant,
    store: auth_store.AuthStore,
    provider: apple.AppleAuthProvider,
) -> AuthManager:
    """Mock manager."""
    return AuthManager(hass, store, {(provider.type, provider.id): provider}, {})


def make_identity_token(
    private_key: rsa.RSAPrivateKey,
    *,
    audience: str = CLIENT_ID,
    issuer: str = apple.APPLE_ISSUER,
    subject: str = "001234.abcdef",
    email: str = "alex@example.com",
    email_verified: bool | str = True,
) -> str:
    """Make a signed Apple-like identity token."""
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "aud": audience,
            "email": email,
            "email_verified": email_verified,
            "exp": now + timedelta(minutes=5),
            "iat": now,
            "iss": issuer,
            "sub": subject,
        },
        private_key,
        algorithm=apple.APPLE_ALGORITHM,
        headers={"kid": KEY_ID},
    )


async def test_not_allow_empty_client_id(hass: HomeAssistant) -> None:
    """Test client id is required."""
    with pytest.raises(vol.Invalid):
        await auth_provider_from_config(hass, None, {"type": "apple", "client_id": ""})


async def test_create_new_credential(
    manager: AuthManager,
    provider: apple.AppleAuthProvider,
    private_key: rsa.RSAPrivateKey,
) -> None:
    """Test that we create a new credential from an Apple identity token."""
    flow_result = await provider.async_validate_identity_token(
        make_identity_token(private_key),
        "Alex",
    )
    credentials = await provider.async_get_or_create_credentials(flow_result)
    assert credentials.is_new is True
    assert credentials.data == {
        "email": "alex@example.com",
        "name": "Alex",
        "subject": "001234.abcdef",
    }

    user = await manager.async_get_or_create_user(credentials)
    assert user.name == "Alex"
    assert user.is_active


async def test_match_existing_credentials(
    provider: apple.AppleAuthProvider,
    private_key: rsa.RSAPrivateKey,
) -> None:
    """Test matching an existing Apple credential by subject."""
    existing = auth_models.Credentials(
        id=uuid.uuid4(),
        auth_provider_type="apple",
        auth_provider_id=None,
        data={"subject": "001234.abcdef"},
        is_new=False,
    )
    provider.async_credentials = AsyncMock(return_value=[existing])

    flow_result = await provider.async_validate_identity_token(
        make_identity_token(private_key)
    )
    credentials = await provider.async_get_or_create_credentials(flow_result)
    assert credentials is existing


async def test_links_verified_email_to_existing_homeassistant_user(
    manager: AuthManager,
    provider: apple.AppleAuthProvider,
    private_key: rsa.RSAPrivateKey,
) -> None:
    """Test a verified Apple email links to an existing Home Assistant user."""
    user = await manager.async_create_user("alex@example.com")
    homeassistant_credentials = auth_models.Credentials(
        auth_provider_type="homeassistant",
        auth_provider_id=None,
        data={"username": "alex@example.com"},
        is_new=False,
    )
    await manager.async_link_user(user, homeassistant_credentials)

    flow_result = await provider.async_validate_identity_token(
        make_identity_token(private_key),
        "Alex",
    )
    credentials = await provider.async_get_or_create_credentials(flow_result)

    assert credentials.is_new is False
    assert credentials.data == {
        "email": "alex@example.com",
        "name": "Alex",
        "subject": "001234.abcdef",
    }
    assert await manager.async_get_or_create_user(credentials) is user
    assert len(user.credentials) == 2


async def test_unverified_email_does_not_link_to_existing_homeassistant_user(
    manager: AuthManager,
    provider: apple.AppleAuthProvider,
    private_key: rsa.RSAPrivateKey,
) -> None:
    """Test an unverified Apple email does not link to an existing user."""
    user = await manager.async_create_user("alex@example.com")
    homeassistant_credentials = auth_models.Credentials(
        auth_provider_type="homeassistant",
        auth_provider_id=None,
        data={"username": "alex@example.com"},
        is_new=False,
    )
    await manager.async_link_user(user, homeassistant_credentials)

    flow_result = await provider.async_validate_identity_token(
        make_identity_token(private_key, email_verified=False),
        "Alex",
    )
    credentials = await provider.async_get_or_create_credentials(flow_result)

    assert credentials.is_new is True
    assert len(user.credentials) == 1


async def test_invalid_audience_rejected(
    provider: apple.AppleAuthProvider,
    private_key: rsa.RSAPrivateKey,
) -> None:
    """Test invalid token audience is rejected."""
    with pytest.raises(apple.InvalidAuthError):
        await provider.async_validate_identity_token(
            make_identity_token(private_key, audience="com.example.OtherApp")
        )


async def test_allowed_subjects_restrict_login(
    hass: HomeAssistant,
    store: auth_store.AuthStore,
    private_key: rsa.RSAPrivateKey,
) -> None:
    """Test allowed subjects restrict who can sign in."""
    provider = apple.AppleAuthProvider(
        hass,
        store,
        {
            "type": "apple",
            "client_id": CLIENT_ID,
            "allowed_subjects": ["expected-subject"],
        },
    )
    provider._async_get_signing_key = AsyncMock(return_value=private_key.public_key())

    with pytest.raises(apple.InvalidAuthError):
        await provider.async_validate_identity_token(make_identity_token(private_key))


async def test_login_flow_invalid_token(provider: apple.AppleAuthProvider) -> None:
    """Test invalid Apple tokens fail the login flow."""
    provider.async_validate_identity_token = AsyncMock(
        side_effect=apple.InvalidAuthError
    )
    flow = await provider.async_login_flow({})
    result = await flow.async_step_init({"identity_token": "not-a-token"})

    assert result["type"] == data_entry_flow.FlowResultType.FORM
    assert result["errors"]["base"] == "invalid_auth"
