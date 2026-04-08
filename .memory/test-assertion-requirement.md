---
name: test-assertion-requirement
description: Tests must include assertions, not just print statements
type: feedback
---

A test function that only contains print() statements without any assert statements is not a valid test. Tests must include assertions to verify expected behavior and provide actual verification value. The verifier explicitly rejected test_mcp_agent in ./rag/tests/test_agent_logic.py because it lacked assertions.
