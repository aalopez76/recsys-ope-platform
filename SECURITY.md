# Security Policy

## Supported Versions

| Version | Supported |
|:--------|:---------:|
| 1.x | ✅ |

## Reporting a Vulnerability

**Do NOT open a public GitHub Issue for security vulnerabilities.**

Please report security issues by emailing: **aalpzp@gmail.com**

Include in the report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We aim to respond within **72 hours** and will keep the reporter informed of progress.

## Scope

This project:
- Processes **no PII** — all user contexts are pre-hashed embeddings from the Open Bandit Dataset.
- Does not expose any network services.
- Is a research/certification codebase, not a production system.

Security concerns most likely relevant to this project:
- Pickle/joblib deserialization of untrusted model files (`saved/`)
- Dependency vulnerabilities in ML stack (tensorflow, scikit-learn, obp)
