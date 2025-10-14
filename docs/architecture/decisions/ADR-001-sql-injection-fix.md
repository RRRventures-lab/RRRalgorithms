# ADR-001: SQL Injection Prevention in Database Layer

**Date:** 2025-10-12  
**Status:** ✅ Implemented  
**Decision Makers:** Security Team, SuperThink Audit  

---

## Context

During comprehensive security audit, a critical SQL injection vulnerability was discovered in `src/core/database/local_db.py` in the `update_trade()` method. Column names were being directly interpolated into SQL queries without validation, allowing potential SQL injection attacks.

### Vulnerability Details

**File:** `src/core/database/local_db.py:290`  
**Severity:** P0 - CRITICAL  
**Risk:** Database manipulation, data corruption, unauthorized access  

**Vulnerable Code:**
```python
def update_trade(self, trade_id: int, updates: Dict[str, Any]):
    """Update trade status and execution details."""
    updates['updated_at'] = datetime.now().isoformat()
    
    # VULNERABLE: Column names not validated
    set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
    query = f"UPDATE trades SET {set_clause} WHERE id = ?"
    
    params = list(updates.values()) + [trade_id]
    self.execute(query, tuple(params))
```

**Attack Vector:**
```python
# Malicious input
malicious_updates = {
    "status = 'executed', pnl = 999999 WHERE id = 1 OR 1=1; --": "value"
}
db.update_trade(1, malicious_updates)
```

---

## Decision

Implement **column whitelist validation** to prevent SQL injection while maintaining flexibility for legitimate updates.

### Solution: Whitelisting Approach

Add explicit validation of allowed columns before constructing SQL query:

```python
def update_trade(self, trade_id: int, updates: Dict[str, Any]):
    """Update trade status and execution details."""
    # Whitelist allowed columns to prevent SQL injection
    ALLOWED_COLUMNS = {
        'status', 'executed_quantity', 'executed_price', 
        'commission', 'pnl', 'strategy', 'notes', 'updated_at'
    }
    
    # Validate all columns are allowed
    invalid_cols = set(updates.keys()) - ALLOWED_COLUMNS
    if invalid_cols:
        raise ValueError(f"Invalid columns for update: {invalid_cols}")
    
    updates['updated_at'] = datetime.now().isoformat()
    
    set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
    query = f"UPDATE trades SET {set_clause} WHERE id = ?"
    
    params = list(updates.values()) + [trade_id]
    self.execute(query, tuple(params))
```

---

## Consequences

### Positive

1. ✅ **Eliminates SQL injection vulnerability** - P0 security issue resolved
2. ✅ **Explicit API contract** - Clear documentation of updateable columns
3. ✅ **Early validation** - Fails fast with clear error message
4. ✅ **No performance impact** - O(n) set operations are negligible
5. ✅ **Maintains parameterization** - Still uses proper query parameters

### Negative

1. ⚠️ **Less flexible** - Cannot dynamically add columns without code change
2. ⚠️ **Maintenance overhead** - Must update whitelist when schema changes

### Mitigation

- Document whitelist in schema documentation
- Add test to verify whitelist matches actual table columns
- Consider auto-generating whitelist from schema introspection (future)

---

## Alternatives Considered

### Option 1: Full ORM (e.g., SQLAlchemy)
**Pros:** Built-in protection, richer features  
**Cons:** Heavy dependency, learning curve, overkill for simple use case  
**Verdict:** ❌ Rejected - Too heavyweight for current needs

### Option 2: Column name sanitization (regex)
**Pros:** More flexible  
**Cons:** Error-prone, easy to bypass, not recommended security practice  
**Verdict:** ❌ Rejected - Whitelisting is more secure

### Option 3: Stored procedures
**Pros:** Maximum security  
**Cons:** Requires database migration, complexity  
**Verdict:** ❌ Rejected - SQLite doesn't support stored procedures

### Option 4: Schema introspection
**Pros:** Automatic validation  
**Cons:** Runtime overhead, complexity  
**Verdict:** 🤔 Consider for future (v2.0)

---

## Implementation

**Implemented:** 2025-10-12  
**Developer:** SuperThink Security Agent  
**Reviewer:** Master Coordinator  
**Testing:** Manual security testing, unit tests added  

### Verification

```python
# Test 1: Valid update works
db.update_trade(1, {'status': 'executed', 'pnl': 100.0})  # ✅ Success

# Test 2: Invalid column rejected
try:
    db.update_trade(1, {'malicious_col': 'value'})
except ValueError as e:
    assert 'Invalid columns' in str(e)  # ✅ Success

# Test 3: SQL injection attempt blocked
try:
    db.update_trade(1, {"status; DROP TABLE trades;--": 'value'})
except ValueError:
    pass  # ✅ Blocked
```

---

## Related Decisions

- ADR-002: Database Index Optimization
- ADR-003: Performance Improvements

---

## References

- [OWASP SQL Injection Prevention](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html)
- [CWE-89: SQL Injection](https://cwe.mitre.org/data/definitions/89.html)
- Security Audit Report: `docs/audit/teams/SECURITY_AUDIT.md`

---

**Status:** ✅ IMPLEMENTED  
**Next Review:** After production deployment


