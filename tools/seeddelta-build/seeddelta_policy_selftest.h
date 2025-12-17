// SeedDelta policy merge self-tests.
// These tests are intended to be lightweight and runnable in CI or locally via
// `llama-seeddelta-build --policy-self-test`.

#pragma once

// Returns 0 on success, non-zero on failure.
int sd_policy_self_test();

