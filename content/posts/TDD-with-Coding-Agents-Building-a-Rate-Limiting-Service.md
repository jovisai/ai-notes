---
title: 'TDD with Coding Agents: Building a Rate Limiting Service'
date: "2025-08-06"
tags: ["AI", "Agentic AI", "LLMs", "Development", "Coding", "Tutorial", "Autonomous Agents"]
---

## Problem Overview

We'll build a sophisticated rate limiting service that supports:
- Multiple rate limiting algorithms (Token Bucket, Fixed Window, Sliding Window)
- Different storage backends (Memory, Redis)
- Per-user and per-API-key limits
- Rate limit headers in responses
- Graceful degradation when storage fails

This is complex enough to demonstrate TDD's power with AI agents.

## Why TDD Works Exceptionally Well with AI Agents

**The AI Agent Advantage:**
- **Perfect Memory**: Never forgets edge cases once written in tests
- **Pattern Recognition**: Excellent at implementing algorithms to match test specifications
- **Systematic Approach**: Follows TDD discipline consistently
- **Rapid Iteration**: Fast feedback cycles between test and implementation

**The Key Insight**: AI agents excel when they have clear specifications (tests) rather than vague requirements.

---

## Phase 1: Setting Up the TDD Environment

### Claude Code Prompt for Setup
```markdown
We're building a rate limiting service using strict TDD. Set up the project structure:

1. Create a Python project with pytest
2. Set up the basic directory structure:
   - src/rate_limiter/
   - tests/
   - requirements.txt
3. Install dependencies: pytest, redis, typing-extensions
4. Create __init__.py files
5. Set up a basic pytest configuration

IMPORTANT: This is TDD - we'll write tests first, then implement. Don't create any implementation code yet, just the project scaffold.
```

Expected project structure:
```
rate_limiter/
├── src/
│   └── rate_limiter/
│       ├── __init__.py
│       ├── core.py (empty for now)
│       ├── algorithms.py (empty for now)
│       └── storage.py (empty for now)
├── tests/
│   ├── __init__.py
│   ├── test_token_bucket.py
│   ├── test_fixed_window.py
│   └── test_integration.py
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Phase 2: RED - Writing Failing Tests First

### Step 1: Core Interface Tests

**Claude Code Prompt:**
```markdown
Now we write our first failing tests. Create comprehensive tests for the core RateLimiter interface.

Requirements to test:
- RateLimiter should check if a request is allowed
- Should return remaining quota and reset time
- Should handle different rate limit rules (requests per minute/hour)
- Should support different identifiers (user_id, api_key, ip_address)

Write these tests in tests/test_core.py. Make them descriptive and cover edge cases:
- Normal request flow
- Exceeding rate limit
- Time-based reset behavior
- Multiple identifiers
- Invalid inputs

Remember: PURE TDD - write tests that describe exactly what we want, but don't implement anything yet. The tests should fail because we haven't written the implementation.
```

**Expected Test File (tests/test_core.py):**
```python
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.rate_limiter.core import RateLimiter, RateLimitResult, RateLimitRule


class TestRateLimiterCore:
    def test_allows_request_within_limit(self):
        """Test that requests within limit are allowed"""
        storage = Mock()
        storage.get_current_count.return_value = 5
        
        limiter = RateLimiter(storage)
        rule = RateLimitRule(limit=10, window_seconds=60)
        
        result = limiter.check_limit("user123", rule)
        
        assert result.allowed is True
        assert result.remaining == 5
        assert result.reset_time is not None
    
    def test_denies_request_when_limit_exceeded(self):
        """Test that requests exceeding limit are denied"""
        storage = Mock()
        storage.get_current_count.return_value = 10
        
        limiter = RateLimiter(storage)
        rule = RateLimitRule(limit=10, window_seconds=60)
        
        result = limiter.check_limit("user123", rule)
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after > 0
    
    def test_resets_count_after_window_expires(self):
        """Test that rate limit resets after time window"""
        storage = Mock()
        # First call: at limit
        # Second call: after reset, should be allowed
        storage.get_current_count.side_effect = [10, 1]
        
        limiter = RateLimiter(storage)
        rule = RateLimitRule(limit=10, window_seconds=60)
        
        # First request - should be denied
        result1 = limiter.check_limit("user123", rule)
        assert result1.allowed is False
        
        # Simulate time passing (mocked in implementation)
        result2 = limiter.check_limit("user123", rule)
        assert result2.allowed is True
    
    def test_handles_different_identifiers(self):
        """Test rate limiting works for different identifier types"""
        storage = Mock()
        storage.get_current_count.return_value = 5
        
        limiter = RateLimiter(storage)
        rule = RateLimitRule(limit=10, window_seconds=60)
        
        # Different identifiers should be tracked separately
        result1 = limiter.check_limit("user123", rule)
        result2 = limiter.check_limit("api_key_abc", rule)
        result3 = limiter.check_limit("192.168.1.1", rule)
        
        assert all(r.allowed for r in [result1, result2, result3])
        # Should have called storage for each identifier
        assert storage.get_current_count.call_count == 3
    
    def test_raises_error_for_invalid_inputs(self):
        """Test proper error handling for invalid inputs"""
        storage = Mock()
        limiter = RateLimiter(storage)
        
        with pytest.raises(ValueError):
            limiter.check_limit("", RateLimitRule(10, 60))  # Empty identifier
        
        with pytest.raises(ValueError):
            limiter.check_limit("user123", RateLimitRule(0, 60))  # Zero limit
        
        with pytest.raises(ValueError):
            limiter.check_limit("user123", RateLimitRule(10, 0))  # Zero window


class TestRateLimitRule:
    def test_creates_valid_rule(self):
        """Test RateLimitRule creation with valid parameters"""
        rule = RateLimitRule(limit=100, window_seconds=3600)
        assert rule.limit == 100
        assert rule.window_seconds == 3600
    
    def test_validates_rule_parameters(self):
        """Test RateLimitRule validation"""
        with pytest.raises(ValueError):
            RateLimitRule(limit=-1, window_seconds=60)
        
        with pytest.raises(ValueError):
            RateLimitRule(limit=10, window_seconds=-1)


class TestRateLimitResult:
    def test_result_contains_required_fields(self):
        """Test RateLimitResult has all required fields"""
        reset_time = datetime.now() + timedelta(seconds=60)
        
        result = RateLimitResult(
            allowed=True,
            remaining=5,
            reset_time=reset_time,
            retry_after=None
        )
        
        assert result.allowed is True
        assert result.remaining == 5
        assert result.reset_time == reset_time
        assert result.retry_after is None
```

### Step 2: Algorithm-Specific Tests

**Claude Code Prompt:**
```markdown
Now create comprehensive tests for the Token Bucket algorithm specifically. This should test:

- Token bucket starts full
- Tokens are consumed on each request  
- Tokens refill at specified rate
- Burst capacity handling
- Time-based token refill
- Edge cases like system clock changes

Create tests/test_token_bucket.py. Again, pure TDD - comprehensive tests but no implementation.
```

**Expected Test File (tests/test_token_bucket.py):**
```python
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.rate_limiter.algorithms import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    def test_allows_requests_when_bucket_has_tokens(self):
        """Test requests are allowed when tokens are available"""
        storage = Mock()
        # Mock storage to return bucket state: 5 tokens, last refill now
        now = datetime.now()
        storage.get_bucket_state.return_value = (5, now.timestamp())
        
        limiter = TokenBucketRateLimiter(storage)
        
        result = limiter.check_limit(
            identifier="user123",
            capacity=10,
            refill_rate=1.0,  # 1 token per second
            requested_tokens=1
        )
        
        assert result.allowed is True
        assert result.remaining == 4  # 5 - 1 requested
    
    def test_denies_request_when_no_tokens_available(self):
        """Test requests are denied when bucket is empty"""
        storage = Mock()
        now = datetime.now()
        storage.get_bucket_state.return_value = (0, now.timestamp())
        
        limiter = TokenBucketRateLimiter(storage)
        
        result = limiter.check_limit(
            identifier="user123",
            capacity=10,
            refill_rate=1.0,
            requested_tokens=1
        )
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after > 0  # Should indicate when tokens will be available
    
    def test_refills_tokens_based_on_time_elapsed(self):
        """Test tokens are refilled based on elapsed time"""
        storage = Mock()
        now = datetime.now()
        past_time = now - timedelta(seconds=10)  # 10 seconds ago
        
        # Bucket had 2 tokens, 10 seconds ago
        # With refill_rate=1.0, should have 12 tokens now (2 + 10*1.0)
        # But capped at capacity=10
        storage.get_bucket_state.return_value = (2, past_time.timestamp())
        
        limiter = TokenBucketRateLimiter(storage)
        
        with patch('src.rate_limiter.algorithms.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            
            result = limiter.check_limit(
                identifier="user123",
                capacity=10,
                refill_rate=1.0,
                requested_tokens=1
            )
        
        assert result.allowed is True
        assert result.remaining == 9  # min(2 + 10*1.0, 10) - 1 = 9
    
    def test_handles_burst_requests_up_to_capacity(self):
        """Test burst requests are handled up to bucket capacity"""
        storage = Mock()
        now = datetime.now()
        storage.get_bucket_state.return_value = (10, now.timestamp())  # Full bucket
        
        limiter = TokenBucketRateLimiter(storage)
        
        # Request 5 tokens at once
        result = limiter.check_limit(
            identifier="user123",
            capacity=10,
            refill_rate=1.0,
            requested_tokens=5
        )
        
        assert result.allowed is True
        assert result.remaining == 5
    
    def test_denies_burst_request_exceeding_available_tokens(self):
        """Test burst requests exceeding available tokens are denied"""
        storage = Mock()
        now = datetime.now()
        storage.get_bucket_state.return_value = (3, now.timestamp())  # 3 tokens available
        
        limiter = TokenBucketRateLimiter(storage)
        
        # Request 5 tokens, but only 3 available
        result = limiter.check_limit(
            identifier="user123",
            capacity=10,
            refill_rate=1.0,
            requested_tokens=5
        )
        
        assert result.allowed is False
        assert result.remaining == 3  # Unchanged
    
    def test_updates_storage_with_new_bucket_state(self):
        """Test storage is updated with new bucket state after request"""
        storage = Mock()
        now = datetime.now()
        storage.get_bucket_state.return_value = (5, now.timestamp())
        
        limiter = TokenBucketRateLimiter(storage)
        
        with patch('src.rate_limiter.algorithms.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            
            limiter.check_limit("user123", 10, 1.0, 1)
        
        # Should update storage with new state: 4 tokens, current timestamp
        storage.set_bucket_state.assert_called_once_with("user123", 4, now.timestamp())
    
    def test_handles_clock_changes_gracefully(self):
        """Test algorithm handles system clock changes"""
        storage = Mock()
        now = datetime.now()
        future_time = now + timedelta(hours=1)  # Clock jumped forward
        
        # Last refill was in the "future" due to clock change
        storage.get_bucket_state.return_value = (5, future_time.timestamp())
        
        limiter = TokenBucketRateLimiter(storage)
        
        with patch('src.rate_limiter.algorithms.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            
            result = limiter.check_limit("user123", 10, 1.0, 1)
        
        # Should handle gracefully, not add negative time
        assert result.allowed is True
        assert result.remaining >= 0
    
    def test_calculates_correct_retry_after_time(self):
        """Test retry_after is calculated correctly when denied"""
        storage = Mock()
        now = datetime.now()
        storage.get_bucket_state.return_value = (0, now.timestamp())
        
        limiter = TokenBucketRateLimiter(storage)
        
        result = limiter.check_limit("user123", 10, 2.0, 1)  # 2 tokens per second
        
        assert result.allowed is False
        # Need 1 token, refill rate is 2/second, so should wait 0.5 seconds
        assert abs(result.retry_after - 0.5) < 0.01
```

### Step 3: Storage Backend Tests

**Claude Code Prompt:**
```markdown
Create tests for Redis and Memory storage backends. Test:

Memory Storage:
- Thread safety for concurrent access
- Proper data isolation between identifiers
- Memory cleanup/expiration

Redis Storage:  
- Connection handling and retries
- Atomic operations for rate limiting
- Graceful fallback when Redis is unavailable
- Proper key expiration

Create tests/test_storage.py with comprehensive coverage.
```

---

## Phase 3: RED Phase Verification

**Claude Code Prompt:**
```markdown
Run the test suite to confirm all tests fail as expected. This is crucial in TDD - we need to see the RED phase.

Execute:
```bash
pytest -v
```

Expected output: All tests should fail with import errors or missing implementations. This proves our tests are actually testing something and not passing accidentally.

If any tests pass, we need to fix them to properly fail first.
```

**Expected Output:**
```
================================== FAILURES ==================================
ImportError: No module named 'src.rate_limiter.core'
ImportError: No module named 'src.rate_limiter.algorithms'
ImportError: No module named 'src.rate_limiter.storage'

=============================== short test summary info ===============================
FAILED tests/test_core.py - ImportError
FAILED tests/test_token_bucket.py - ImportError  
FAILED tests/test_storage.py - ImportError
```

---

## Phase 4: GREEN - Making Tests Pass (Minimal Implementation)

### Step 1: Core Implementation

**Claude Code Prompt:**
```markdown
Now implement the MINIMUM code needed to make the core tests pass. Follow TDD strictly:

1. Create the core classes and interfaces in src/rate_limiter/core.py
2. Implement only what's needed for tests to pass
3. Don't add any extra functionality beyond what tests require
4. Focus on making tests green, not on perfect code

Start with the data structures and basic RateLimiter class.
```

**Expected Implementation (src/rate_limiter/core.py):**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class RateLimitRule:
    limit: int
    window_seconds: int
    
    def __post_init__(self):
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        if self.window_seconds <= 0:
            raise ValueError("Window seconds must be positive")


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_time: Optional[datetime] = None
    retry_after: Optional[float] = None


class RateLimitStorage(ABC):
    @abstractmethod
    def get_current_count(self, identifier: str, rule: RateLimitRule) -> int:
        pass
    
    @abstractmethod
    def increment_count(self, identifier: str, rule: RateLimitRule) -> int:
        pass


class RateLimiter:
    def __init__(self, storage: RateLimitStorage):
        self.storage = storage
    
    def check_limit(self, identifier: str, rule: RateLimitRule) -> RateLimitResult:
        if not identifier:
            raise ValueError("Identifier cannot be empty")
        
        current_count = self.storage.get_current_count(identifier, rule)
        
        if current_count >= rule.limit:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                retry_after=60.0  # Simplified - just return window seconds
            )
        
        # Allow request and increment
        new_count = self.storage.increment_count(identifier, rule)
        remaining = rule.limit - new_count
        
        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_time=datetime.now() + timedelta(seconds=rule.window_seconds)
        )
```

### Step 2: Token Bucket Implementation

**Claude Code Prompt:**
```markdown
Now implement the TokenBucketRateLimiter to make those tests pass. Remember - minimal implementation that satisfies the tests, we'll refactor later.

Key requirements from tests:
- Token bucket with capacity and refill rate
- Time-based token refill
- Burst request handling
- Storage integration
- Clock change handling
```

### Step 3: Iterative Implementation

**Claude Code Prompt for each iteration:**
```markdown
Run the tests again:
```bash
pytest tests/test_core.py -v
```

Fix any failing tests one by one. For each failure:
1. Understand what the test expects
2. Implement the minimal change to make it pass
3. Don't optimize yet - just make it work
4. Move to the next failing test

Show me each test result and the code changes needed.
```

---

## Phase 5: GREEN Phase Completion

Continue this process:

1. **Run tests**
2. **See failures**  
3. **Implement minimal fixes**
4. **Repeat until all tests pass**

**Key TDD Principle**: Don't write more code than needed to pass the tests.

---

## Phase 6: REFACTOR - Improve Code Quality

Once all tests are green:

**Claude Code Prompt:**
```markdown
All tests are now passing. Time for the REFACTOR phase. Improve the code quality while keeping all tests green:

1. Extract common patterns into helper methods
2. Improve error handling and edge cases
3. Add proper logging and monitoring hooks
4. Optimize performance bottlenecks
5. Improve code readability and documentation

Run tests after each refactoring to ensure they stay green. The key rule: improve code without changing behavior.
```

**Refactoring Examples:**
- Extract time calculations into utility methods
- Add proper logging for debugging
- Implement connection pooling for Redis
- Add configuration management
- Improve error messages
- Add type hints and documentation

---

## Phase 7: Integration Tests

**Claude Code Prompt:**
```markdown
Now write integration tests that test the complete system working together:

1. Real Redis backend with test containers
2. Multiple rate limiting algorithms working together  
3. Concurrent request handling
4. Performance under load
5. Failure scenarios (Redis down, network issues)

These are higher-level tests that ensure our components work together correctly.
```

---

## Why This TDD Approach Works So Well with AI Agents

### 1. **Clear Specifications**
```markdown
# Instead of vague requirements:
"Build a rate limiter"

# TDD gives clear specifications:
"test_denies_request_when_limit_exceeded should fail when current_count >= limit"
```

### 2. **Systematic Progress**
The AI agent follows a methodical approach:
- Red → Green → Refactor → Red → Green → Refactor
- Never skips steps or adds unnecessary features
- Focuses on exactly what tests require

### 3. **Perfect Memory for Edge Cases**
Once written in tests, the AI never forgets:
- Clock changes
- Concurrent access
- Error conditions  
- Boundary conditions

### 4. **Rapid Iteration**
AI agents excel at the fast feedback cycle:
- Write test → Run → Fix → Repeat
- No fatigue or rushing through phases
- Consistent discipline

---

## Advanced TDD Techniques with AI Agents

### Property-Based Testing
```markdown
"Add property-based tests using hypothesis library:
- Rate limiter should never allow more requests than the limit
- Token bucket should never have negative tokens
- Time-based calculations should be monotonic"
```

### Test Data Builders
```markdown
"Create test data builders for complex scenarios:
- RateLimitRuleBuilder for different rule types
- MockStorageBuilder for various storage states  
- ScenarioBuilder for integration test cases"
```

### Mutation Testing
```markdown
"Use mutmut to verify our tests catch all possible bugs:
1. Run mutation testing on the rate limiter code
2. Identify any surviving mutants
3. Add tests to kill remaining mutants"
```

---

## Common TDD Pitfalls to Avoid with AI Agents

### 1. **Writing Too Much Code**
```markdown
❌ "Implement a complete rate limiter with all features"

✅ "Implement only what's needed to make test_allows_request_within_limit pass"
```

### 2. **Skipping the Red Phase**
```markdown
❌ Writing tests after implementation

✅ "Write the test first, run it to confirm it fails, then implement"
```

### 3. **Testing Implementation Details**
```markdown
❌ Testing internal method calls

✅ Testing public behavior and outcomes
```

### 4. **Not Refactoring**
```markdown
❌ Leaving code messy once tests pass

✅ "Clean up this code while keeping all tests green"
```

---

## Measuring Success

### Test Quality Metrics
- **Coverage**: Aim for 100% branch coverage
- **Mutation Score**: Use mutation testing to verify test quality
- **Test Speed**: Fast feedback loops (< 1 second)

### Code Quality Metrics  
- **Cyclomatic Complexity**: Keep methods simple
- **Code Duplication**: Extract common patterns
- **Documentation**: Tests serve as living documentation

### TDD Process Metrics
- **Red-Green-Refactor Cycles**: Track discipline adherence
- **Test-First Percentage**: Measure how often tests are written first
- **Refactoring Frequency**: Ensure regular code improvement

---

## Conclusion

TDD with AI agents is incredibly powerful because:

1. **AI agents excel with clear specifications** (tests)
2. **They maintain discipline** in the Red-Green-Refactor cycle
3. **Perfect memory** ensures edge cases are never forgotten
4. **Rapid iteration** enables fast feedback loops
5. **Systematic approach** prevents feature creep

The combination creates a development experience where you focus on **what** the system should do (tests) and let the AI figure out **how** to implement it efficiently.

Start with simple examples, build confidence in the process, then tackle increasingly complex problems. The rate limiter example shows how even sophisticated systems become manageable when broken down into testable components.

**Next Steps:**
1. Try this TDD approach on a simpler problem first
2. Practice the Red-Green-Refactor discipline
3. Gradually increase complexity
4. Share your experiences with the community

Remember: TDD isn't about testing - it's about **design through examples**. The tests become your specification, and the AI agent becomes your implementation partner.