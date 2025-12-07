# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

**Do NOT open a public issue for security vulnerabilities.**

Instead, please email the maintainers directly with:
1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Any suggested fixes (optional)

We will respond within 48 hours and work with you to understand and address the issue.

---

## Security Considerations

### Differential Privacy Guarantees

This project implements differential privacy with the following considerations:

#### What DP Protects Against
- ✅ Membership inference attacks (detecting if a record was in training data)
- ✅ Training data extraction attacks
- ✅ Model inversion attacks
- ✅ Attribute inference for training records

#### What DP Does NOT Protect Against
- ❌ Attacks on the redaction pipeline itself
- ❌ Side-channel attacks (timing, power analysis)
- ❌ Attacks exploiting bugs in implementation
- ❌ Social engineering

### Privacy Budget Selection

| Use Case | Recommended ε | Notes |
|----------|---------------|-------|
| Highly sensitive data (medical, financial) | 1.0 - 2.0 | Prioritize privacy over accuracy |
| Standard business use | 3.0 - 5.0 | Balanced tradeoff |
| Low-sensitivity applications | 5.0 - 8.0 | Prioritize accuracy |

**Warning:** ε > 10 provides minimal privacy guarantees and should not be relied upon.

### Data Handling

1. **Training Data**
   - Use synthetic or properly anonymized data when possible
   - Ensure data is stored securely during training
   - Delete intermediate checkpoints containing gradient information

2. **Model Artifacts**
   - Trained models may retain some information despite DP
   - Do not share models trained on sensitive data without review
   - Consider model pruning to reduce potential memorization

3. **Redaction Output**
   - Verify redaction quality before sharing documents
   - False negatives (missed PII) are possible
   - Human review recommended for highly sensitive documents

---

## Secure Deployment Checklist

### Before Deployment

- [ ] Verify ε value is appropriate for sensitivity level
- [ ] Test redaction on representative samples
- [ ] Review false negative rate
- [ ] Ensure model files are not publicly accessible
- [ ] Set up access logging for audit purposes

### During Operation

- [ ] Monitor for unusual patterns in usage
- [ ] Regularly update dependencies for security patches
- [ ] Log redaction requests (without storing original PII)
- [ ] Implement rate limiting to prevent abuse

### Access Control

```python
# Example: Restrict access to redaction API
from functools import wraps

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            raise PermissionError("Authentication required")
        return f(*args, **kwargs)
    return decorated

@require_auth
def redact_document(text):
    return redactor.redact(text)
```

---

## Dependency Security

### Known Vulnerabilities

We regularly scan dependencies. Check for updates:

```bash
pip install safety
safety check -r requirements.txt
```

### Pinned Versions

For production, use pinned versions:

```bash
pip freeze > requirements-lock.txt
```

---

## Responsible AI Considerations

### Bias and Fairness

- The model may have different performance across demographic groups
- Test on diverse data before deployment
- Monitor for disparate impact in production

### Misuse Prevention

This tool should NOT be used to:
- Evade legitimate privacy regulations
- Process data without proper consent
- Circumvent data subject rights

---

## Security Updates

| Date | Version | Issue | Resolution |
|------|---------|-------|------------|
| - | - | No known vulnerabilities | - |

---

## Contact

For security concerns, contact the maintainers:
- Lanre Atoye - University of Guelph
- Thomas Martial - University of Guelph
