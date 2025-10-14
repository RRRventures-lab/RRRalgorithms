# API Key Rotation Guide

## CRITICAL SECURITY INCIDENT

**Date**: 2025-10-11
**Status**: ACTIVE SECURITY BREACH
**Severity**: CRITICAL

All API keys in `config/api-keys/.env` have been exposed in plaintext and MUST be rotated immediately before any live trading deployment.

---

## Priority Order (Complete in this order)

### 1. HIGHEST PRIORITY: Coinbase Exchange API
**Risk**: Direct financial access - unauthorized trades possible
**Action**: Rotate IMMEDIATELY

#### Steps:
1. Log in to [Coinbase Developer Platform](https://portal.cdp.coinbase.com/)
2. Navigate to API Keys section
3. **REVOKE** the existing API key:
   - Organization: `bd2a5d9b-3749-4069-9045-8e48316621ed`
   - Key: `e8d1f3af-946a-43fa-8c94-6c4167a54c3e`
4. Create NEW API key with minimum required permissions:
   - Read permissions: wallet:accounts:read, wallet:transactions:read
   - Trade permissions: Only if absolutely necessary for your use case
   - DO NOT enable withdraw permissions
5. Download the new private key (EC PRIVATE KEY)
6. Update `.env` with new credentials
7. Test connection with paper trading first

**Documentation**: https://docs.cdp.coinbase.com/developer-platform/docs/api-keys

---

### 2. HIGH PRIORITY: Anthropic Claude API
**Risk**: Unauthorized AI usage, potential data exposure through prompts
**Action**: Rotate within 24 hours

#### Steps:
1. Log in to [Anthropic Console](https://console.anthropic.com/)
2. Navigate to API Keys section
3. **DELETE** the exposed key (starts with `sk-ant-api03-`)
4. Create NEW API key
5. Update `.env` with new key
6. Test with a simple API call

**Documentation**: https://docs.anthropic.com/claude/docs/getting-started-with-the-api

---

### 3. HIGH PRIORITY: GitHub Personal Access Token
**Risk**: Unauthorized repository access, code theft, malicious commits
**Action**: Rotate within 24 hours

#### Steps:
1. Log in to GitHub
2. Go to Settings → Developer settings → [Personal access tokens](https://github.com/settings/tokens)
3. Find and **DELETE** token `<REDACTED_GITHUB_PAT>`
4. Generate NEW token (classic)
5. Required scopes:
   - `repo` (Full control of private repositories)
   - `read:org` (Read org membership)
6. Copy and save token immediately (you won't see it again)
7. Update `.env` and any CI/CD configurations

**Documentation**: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens

---

### 4. MEDIUM PRIORITY: Supabase Database
**Risk**: Data breach, unauthorized database access
**Action**: Rotate within 48 hours

#### Steps:
1. Log in to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select project: `isqznbvfmjmghxvctguh`
3. Go to Settings → Database
4. **Reset database password**:
   - Current password: `<REDACTED_SUPABASE_PASSWORD>`
   - Generate new strong password (32+ characters)
5. Update connection strings in `.env`:
   - `SUPABASE_DB_URL`
   - `DATABASE_URL`
6. Go to Settings → API
7. **Rotate service_role key** (if you have one configured)
8. Note: The `anon` key is public by design, but consider RLS policies
9. Test database connection

**Documentation**: https://supabase.com/docs/guides/database/managing-passwords

---

### 5. MEDIUM PRIORITY: Polygon.io Market Data
**Risk**: Unauthorized data usage, billing fraud
**Action**: Rotate within 48 hours

#### Steps:
1. Log in to [Polygon.io Dashboard](https://polygon.io/dashboard)
2. Navigate to API Keys section
3. **Delete** or **deactivate** key: `<REDACTED_POLYGON_KEY>`
4. Create NEW API key
5. Update `.env` with new key
6. Test market data endpoints

**Documentation**: https://polygon.io/docs/getting-started

---

### 6. MEDIUM PRIORITY: Perplexity AI
**Risk**: Unauthorized AI usage, billing fraud
**Action**: Rotate within 48 hours

#### Steps:
1. Log in to [Perplexity AI Dashboard](https://www.perplexity.ai/settings/api)
2. Navigate to API Keys section
3. **Revoke** key: `<REDACTED_PERPLEXITY_KEY>`
4. Create NEW API key
5. Update `.env` with new key
6. Test API connection

**Documentation**: https://docs.perplexity.ai/docs/getting-started

---

### 7. LOW PRIORITY: JWT Secret & Encryption Key
**Risk**: Session hijacking, data decryption (internal only)
**Action**: Rotate before production deployment

#### Steps:
1. Generate new JWT secret:
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(64))"
   ```
2. Generate new encryption key (32 bytes):
   ```bash
   python3 -c "import secrets; print(secrets.token_hex(32))"
   ```
3. Update `.env` with new values
4. **WARNING**: This will invalidate all existing sessions and encrypted data
5. Plan downtime or migration strategy

---

## Verification Checklist

After rotating each key, verify:

- [ ] **Coinbase**: Test paper trading connection
- [ ] **Anthropic**: Make test API call
- [ ] **GitHub**: Clone repo with new token
- [ ] **Supabase**: Connect to database and query
- [ ] **Polygon**: Fetch market data
- [ ] **Perplexity**: Make test search query
- [ ] **JWT/Encryption**: Test internal APIs

---

## Post-Rotation Actions

1. **Update all deployment environments**:
   - Development servers
   - Staging servers
   - CI/CD pipelines
   - Docker containers
   - Kubernetes secrets

2. **Update team members**:
   - Notify all developers
   - Update shared documentation
   - Revoke old credentials from password managers

3. **Enable secrets management**:
   - Use the automated secrets management system (see SECRETS_MANAGEMENT.md)
   - Never store plaintext credentials again

4. **Set up monitoring**:
   - Enable API key usage alerts
   - Monitor for suspicious activity
   - Set up billing alerts

---

## Emergency Contacts

If you detect unauthorized usage:

1. **Coinbase**: Immediately freeze account via [support](https://help.coinbase.com/)
2. **Financial accounts**: Contact your bank
3. **GitHub**: Enable 2FA and review audit log
4. **Other services**: Revoke keys and contact support

---

## Prevention Measures

To prevent future exposures:

1. **Use secrets management**: Migrate to macOS Keychain or similar
2. **Enable .gitignore**: Ensure `.env` is never committed
3. **Use environment variables**: Different keys per environment
4. **Implement rotation policy**: Rotate keys every 90 days
5. **Audit regularly**: Weekly security audits
6. **Principle of least privilege**: Minimal permissions for each key
7. **Enable 2FA**: On all services that support it

---

**Last Updated**: 2025-10-11
**Next Review**: After all keys rotated
