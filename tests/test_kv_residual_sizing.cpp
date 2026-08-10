// kv_blocks_from_residual — the arithmetic that sizes the KV pool from what is
// left after the weight caches are built.
//
// This is a CPU-lane test on purpose. The decision it covers is pure integer
// arithmetic, but until it lived inline in engine_kv_cache_init.cpp,
// where nothing could reach it without a GPU and a 22 GB checkpoint — so the
// one case that matters was never asserted anywhere.
//
// The case that matters: the caller subtracts the allocator headroom from the
// measured residual. If whoever reserved room *for* this pool did not also
// reserve the headroom, the residual is entirely headroom, `room` is 0, and the
// pool silently falls to the floor. That is a rescue, not a size, and
// `floored` is what tells the two apart.

#include <gtest/gtest.h>

#include "memory/vram_query.h"

namespace quench {
namespace {

constexpr size_t kMiB = 1024ULL * 1024ULL;

// Qwen3.6-35B-A3B-UD-Q4_K_M on a 32 GiB 5090: 10 attention layers, block_size
// 32, 2 KV heads, head_dim 256, FP16 -> 640 KiB per block.
constexpr size_t k35bPerBlock = 640ULL * 1024ULL;

TEST(KvResidualSizing, PlanFitsSoNothingIsClamped) {
    const auto s = kv_blocks_from_residual(8192 * kMiB, 1630 * kMiB, 0, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 4096);
    EXPECT_FALSE(s.clamped);
    EXPECT_FALSE(s.floored);
}

TEST(KvResidualSizing, ResidualSmallerThanPlanClampsButDoesNotFloor) {
    // 2416 MiB free - 1630 MiB headroom = 786 MiB -> 1257 blocks. These are the
    // post-fix numbers measured on the repro.
    const auto s = kv_blocks_from_residual(2416 * kMiB, 1630 * kMiB, 0, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 1257);
    EXPECT_TRUE(s.clamped);
    EXPECT_FALSE(s.floored) << "a pool smaller than the plan is normal — the plan is a "
                               "projection, the residual is the truth";
}

// The regression itself, with the numbers straight out of the bug report.
TEST(KvResidualSizing, HeadroomExceedingResidualFloorsAndSaysSo) {
    const auto s = kv_blocks_from_residual(1264 * kMiB, 1630 * kMiB, 0, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 16) << "the floor, not a computed size";
    EXPECT_TRUE(s.clamped);
    EXPECT_TRUE(s.floored) << ": nothing was left to size the pool from, and the load "
                              "reported success anyway";
}

TEST(KvResidualSizing, ResidualExactlyHeadroomLeavesNothing) {
    const auto s = kv_blocks_from_residual(1630 * kMiB, 1630 * kMiB, 0, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 16);
    EXPECT_TRUE(s.floored);
}

// One block short of the floor still floors; exactly the floor does not.
TEST(KvResidualSizing, FloorBoundaryIsExact) {
    const size_t headroom = 1000 * kMiB;
    const auto below = kv_blocks_from_residual(headroom + 15 * k35bPerBlock, headroom, 0,
                                               k35bPerBlock, 4096, 16);
    EXPECT_EQ(below.blocks, 16);
    EXPECT_TRUE(below.floored);

    const auto at = kv_blocks_from_residual(headroom + 16 * k35bPerBlock, headroom, 0,
                                            k35bPerBlock, 4096, 16);
    EXPECT_EQ(at.blocks, 16);
    EXPECT_TRUE(at.clamped);
    EXPECT_FALSE(at.floored) << "16 blocks that were actually computed is not the rescue path";
}

TEST(KvResidualSizing, MoreRoomThanPlannedNeverGrowsThePool) {
    const auto s = kv_blocks_from_residual(30000 * kMiB, 1630 * kMiB, 0, k35bPerBlock, 64, 16);
    EXPECT_EQ(s.blocks, 64) << "the residual can only shrink the plan, never grow it";
    EXPECT_FALSE(s.clamped);
}

TEST(KvResidualSizing, DegenerateInputsAreInert) {
    const auto zero_block = kv_blocks_from_residual(8192 * kMiB, 1630 * kMiB, 0, 0, 4096, 16);
    EXPECT_EQ(zero_block.blocks, 4096) << "per_block==0 must not divide";
    EXPECT_FALSE(zero_block.clamped);

    const auto no_plan = kv_blocks_from_residual(8192 * kMiB, 1630 * kMiB, 0, k35bPerBlock, 0, 16);
    EXPECT_EQ(no_plan.blocks, 0);
    EXPECT_FALSE(no_plan.floored);
}

TEST(KvResidualSizing, FreeBelowHeadroomDoesNotUnderflow) {
    // free < headroom must saturate at 0 room, not wrap around size_t.
    const auto s = kv_blocks_from_residual(100 * kMiB, 1630 * kMiB, 0, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 16);
    EXPECT_TRUE(s.floored);
}

// ── the pending library reserve ───────────────────────────────────────────
//
// Mistral-Nemo-12B Q8_0 on a 32 GiB 5090, numbers straight out of the
// incident: 40 KV layers, 8 KV heads, head_dim 128, FP16, block_size 16
// -> 2.5 MiB per block across layers. After the weight caches 11305 MiB
// were free, the allocator headroom was 1605 MiB, and cuBLAS/CUTLASS's
// 2297 MiB first-forward claim was still pending. The clamp used to ignore
// the pending claim, sized the pool to 3880 blocks, and the first forward
// ran the card to 2 MiB free — warmup graph capture OOM'd and every later
// kernel launch failed on the poisoned context.

constexpr size_t kNemoPerBlock = 2560ULL * 1024ULL;  // 40 layers x 64 KiB

TEST(KvResidualSizing, PendingLibraryReserveIsKeptOutOfThePool) {
    const auto s =
        kv_blocks_from_residual(11305 * kMiB, 1605 * kMiB, 2297 * kMiB, kNemoPerBlock, 5860, 16);
    // (11305 - 1605 - 2297) MiB / 2.5 MiB = 2961 blocks. Headroom alone
    // said 3880 — the extra 919 blocks were the library reserve.
    EXPECT_EQ(s.blocks, 2961);
    EXPECT_TRUE(s.clamped);
    EXPECT_FALSE(s.floored);
}

TEST(KvResidualSizing, PendingReserveExceedingFreeFloorsAndSaysSo) {
    // Free covers the headroom but not the pending claim on top: sizing from
    // it would hand the first forward a deficit, so the floor is right.
    const auto s =
        kv_blocks_from_residual(2000 * kMiB, 1605 * kMiB, 2297 * kMiB, kNemoPerBlock, 5860, 16);
    EXPECT_EQ(s.blocks, 16);
    EXPECT_TRUE(s.clamped);
    EXPECT_TRUE(s.floored);
}

// ── kv_pool_verdict ───────────────────────────────────────────────────────
//
// The floor case has been loud. This covers the case that stayed
// quiet: a pool that is a real size and still cannot admit one full-length
// request, which the load reports as success.

TEST(KvPoolVerdict, PoolHoldingAFullSequenceIsSufficient) {
    // 4096 blocks x 32 tokens = 131072 tokens against a 8192-token request.
    const auto s = kv_blocks_from_residual(8192 * kMiB, 1630 * kMiB, 0, k35bPerBlock, 4096, 16);
    EXPECT_EQ(kv_pool_verdict(s, 8192, 32), KvPoolVerdict::Sufficient);
}

TEST(KvPoolVerdict, ClampedButStillShortOfOneSequenceIsReported) {
    // 2416 MiB residual - 1630 MiB headroom = 786 MiB -> 1257 blocks, well
    // above the 16-block floor, so `floored` is false and the pre- code
    // said nothing. 1257 x 32 = 40224 tokens; a 65536-token max_seq_len needs
    // 2048 blocks, so no full-length request can ever be admitted.
    const auto s = kv_blocks_from_residual(2416 * kMiB, 1630 * kMiB, 0, k35bPerBlock, 4096, 16);
    ASSERT_TRUE(s.clamped);
    ASSERT_FALSE(s.floored) << "this case must not be the floor, or it is the other message";
    EXPECT_EQ(kv_pool_verdict(s, 65536, 32), KvPoolVerdict::ShortOfOneSequence);
    // Same pool, a request it can serve: nothing to report.
    EXPECT_EQ(kv_pool_verdict(s, 32768, 32), KvPoolVerdict::Sufficient);
}

TEST(KvPoolVerdict, FlooredKeepsItsOwnMessage) {
    // Floored is also short of one sequence, but it has a message of its own
    // that names the missing residual — reporting both would say it twice.
    const auto s = kv_blocks_from_residual(1264 * kMiB, 1630 * kMiB, 0, k35bPerBlock, 4096, 16);
    ASSERT_TRUE(s.floored);
    EXPECT_EQ(kv_pool_verdict(s, 65536, 32), KvPoolVerdict::Floored);
}

TEST(KvPoolVerdict, ExactlyOneSequenceIsSufficient) {
    // The boundary decides whether a pool sized to exactly the request is
    // called a fault. It is not: one sequence fits.
    KvResidualSizing s;
    s.blocks = 64;
    EXPECT_EQ(kv_pool_verdict(s, 2048, 32), KvPoolVerdict::Sufficient);
    s.blocks = 63;
    EXPECT_EQ(kv_pool_verdict(s, 2048, 32), KvPoolVerdict::ShortOfOneSequence);
}

TEST(KvPoolVerdict, PartialTrailingBlockCounts) {
    // 2049 tokens at block_size 32 needs 65 blocks, not 64.
    EXPECT_EQ(kv_blocks_per_sequence(2049, 32), 65);
    EXPECT_EQ(kv_blocks_per_sequence(2048, 32), 64);
    KvResidualSizing s;
    s.blocks = 64;
    EXPECT_EQ(kv_pool_verdict(s, 2049, 32), KvPoolVerdict::ShortOfOneSequence);
}

TEST(KvPoolVerdict, UnsetRequirementIsNotAFault) {
    // No max_seq_len / no block size means there is nothing to check against;
    // the verdict must not turn that into a warning on every load.
    KvResidualSizing s;
    s.blocks = 1;
    EXPECT_EQ(kv_blocks_per_sequence(0, 32), 0);
    EXPECT_EQ(kv_blocks_per_sequence(2048, 0), 0);
    EXPECT_EQ(kv_pool_verdict(s, 0, 32), KvPoolVerdict::Sufficient);
    EXPECT_EQ(kv_pool_verdict(s, 2048, 0), KvPoolVerdict::Sufficient);
}

}  // namespace
}  // namespace quench
