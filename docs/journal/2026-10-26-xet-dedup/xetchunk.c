/* Xet's content-defined chunker, ported exactly from xet-core v1.6.0 (de71453).
 *
 * Source: xet_data/src/deduplication/chunking.rs (Chunker::new, next_boundary), with
 * TARGET_CHUNK_SIZE = 64 KiB, MINIMUM_CHUNK_DIVISOR = 8, MAXIMUM_CHUNK_MULTIPLIER = 2
 * (xet_core_structures/src/xorb_object/constants.rs:5,10,14), and the gear hash of the
 * gearhash crate 0.1.3 (Cargo.lock; crate sha256 c8cf82cf...d213): its scalar
 * next_match (src/scalar.rs: hash = (hash << 1) + table[byte], a match when
 * hash & mask == 0, returning the bytes consumed) and DEFAULT_TABLE (src/table.rs),
 * copied below.
 *
 * The rules, as next_boundary applies them to a chunk starting at `pos`:
 *  - mask = (target - 1) << leading_zeros(target - 1): the top 16 bits for 64 KiB;
 *  - the hash restarts at 0 for each chunk (set_hash(0) after every cut);
 *  - the first min - 64 - 1 bytes are skipped unhashed (min = target / 8 = 8 KiB);
 *  - a match is a boundary only once the chunk is at least min bytes (else hashing
 *    continues);
 *  - a chunk is cut at max = 2 * target = 128 KiB if no match came first;
 *  - the last chunk is whatever remains.
 * The streaming chunker (blocks fed one after another, its chunkbuf carried over) cuts
 * where this whole-buffer form does: xet-core's own test_chunk_boundaries checks that
 * for blocks of 1, 37 and 255 bytes; test_xet.py checks this port on xet-core's vectors
 * and against hf_xet's own file hashes.
 */
#include <stddef.h>
#include <stdint.h>

static const uint64_t GEAR[256] = {
    0xb088d3a9e840f559ULL, 0x5652c7f739ed20d6ULL, 0x45b28969898972abULL, 0x6b0a89d5b68ec777ULL,
    0x368f573e8b7a31b7ULL, 0x1dc636dce936d94bULL, 0x207a4c4e5554d5b6ULL, 0xa474b34628239acbULL,
    0x3b06a83e1ca3b912ULL, 0x90e78d6c2f02baf7ULL, 0xe1c92df7150d9a8aULL, 0x8e95053a1086d3adULL,
    0x5a2ef4f1b83a0722ULL, 0xa50fac949f807faeULL, 0x0e7303eb80d8d681ULL, 0x99b07edc1570ad0fULL,
    0x689d2fb555fd3076ULL, 0x00005082119ea468ULL, 0xc4b08306a88fcc28ULL, 0x3eb0678af6374afdULL,
    0xf19f87ab86ad7436ULL, 0xf2129fbfbe6bc736ULL, 0x481149575c98a4edULL, 0x0000010695477bc5ULL,
    0x1fba37801a9ceaccULL, 0x3bf06fd663a49b6dULL, 0x99687e9782e3874bULL, 0x79a10673aa50d8e3ULL,
    0xe4accf9e6211f420ULL, 0x2520e71f87579071ULL, 0x2bd5d3fd781a8a9bULL, 0x00de4dcddd11c873ULL,
    0xeaa9311c5a87392fULL, 0xdb748eb617bc40ffULL, 0xaf579a8df620bf6fULL, 0x86a6e5da1b09c2b1ULL,
    0xcc2fc30ac322a12eULL, 0x355e2afec1f74267ULL, 0x2d99c8f4c021a47bULL, 0xbade4b4a9404cfc3ULL,
    0xf7b518721d707d69ULL, 0x3286b6587bf32c20ULL, 0x0000b68886af270cULL, 0xa115d6e4db8a9079ULL,
    0x484f7e9c97b2e199ULL, 0xccca7bb75713e301ULL, 0xbf2584a62bb0f160ULL, 0xade7e813625dbcc8ULL,
    0x000070940d87955aULL, 0x8ae69108139e626fULL, 0xbd776ad72fde38a2ULL, 0xfb6b001fc2fcc0cfULL,
    0xc7a474b8e67bc427ULL, 0xbaf6f11610eb5d58ULL, 0x09cb1f5b6de770d1ULL, 0xb0b219e6977d4c47ULL,
    0x00ccbc386ea7ad4aULL, 0xcc849d0adf973f01ULL, 0x73a3ef7d016af770ULL, 0xc807d2d386bdbdfeULL,
    0x7f2ac9966c791730ULL, 0xd037a86bc6c504daULL, 0xf3f17c661eaa609dULL, 0xaca626b04daae687ULL,
    0x755a99374f4a5b07ULL, 0x90837ee65b2caedeULL, 0x6ee8ad93fd560785ULL, 0x0000d9e11053edd8ULL,
    0x9e063bb2d21cdbd7ULL, 0x07ab77f12a01d2b2ULL, 0xec550255e6641b44ULL, 0x78fb94a8449c14c6ULL,
    0xc7510e1bc6c0f5f5ULL, 0x0000320b36e4cae3ULL, 0x827c33262c8b1a2dULL, 0x14675f0b48ea4144ULL,
    0x267bd3a6498decebULL, 0xf1916ff982f5035eULL, 0x86221b7ff434fb88ULL, 0x9dbecee7386f49d8ULL,
    0xea58f8cac80f8f4aULL, 0x008d198692fc64d8ULL, 0x6d38704fbabf9a36ULL, 0xe032cb07d1e7be4cULL,
    0x228d21f6ad450890ULL, 0x635cb1bfc02589a5ULL, 0x4620a1739ca2ce71ULL, 0xa7e7dfe3aae5fb58ULL,
    0x0c10ca932b3c0debULL, 0x2727fee884afed7bULL, 0xa2df1c6df9e2ab1fULL, 0x4dcdd1ac0774f523ULL,
    0x000070ffad33e24eULL, 0xa2ace87bc5977816ULL, 0x9892275ab4286049ULL, 0xc2861181ddf18959ULL,
    0xbb9972a042483e19ULL, 0xef70cd3766513078ULL, 0x00000513abfc9864ULL, 0xc058b61858c94083ULL,
    0x09e850859725e0deULL, 0x9197fb3bf83e7d94ULL, 0x7e1e626d12b64bceULL, 0x520c54507f7b57d1ULL,
    0xbee1797174e22416ULL, 0x6fd9ac3222e95587ULL, 0x0023957c9adfbf3eULL, 0xa01c7d7e234bbe15ULL,
    0xaba2c758b8a38cbbULL, 0x0d1fa0ceec3e2b30ULL, 0x0bb6a58b7e60b991ULL, 0x4333dd5b9fa26635ULL,
    0xc2fd3b7d4001c1a3ULL, 0xfb41802454731127ULL, 0x65a56185a50d18cbULL, 0xf67a02bd8784b54fULL,
    0x696f11dd67e65063ULL, 0x00002022fca814abULL, 0x8cd6be912db9d852ULL, 0x695189b6e9ae8a57ULL,
    0xee9453b50ada0c28ULL, 0xd8fc5ea91a78845eULL, 0xab86bf191a4aa767ULL, 0x0000c6b5c86415e5ULL,
    0x267310178e08a22eULL, 0xed2d101b078bca25ULL, 0x3b41ed84b226a8fbULL, 0x13e622120f28dc06ULL,
    0xa315f5ebfb706d26ULL, 0x8816c34e3301baceULL, 0xe9395b9cbb71fdaeULL, 0x002ce9202e721648ULL,
    0x4283db1d2bb3c91cULL, 0xd77d461ad2b1a6a5ULL, 0xe2ec17e46eeb866bULL, 0xb8e0be4039fbc47cULL,
    0xdea160c4d5299d04ULL, 0x7eec86c8d28c3634ULL, 0x2119ad129f98a399ULL, 0xa6ccf46b61a283efULL,
    0x2c52cedef658c617ULL, 0x2db4871169acdd83ULL, 0x0000f0d6f39ecbe9ULL, 0x3dd5d8c98d2f9489ULL,
    0x8a1872a22b01f584ULL, 0xf282a4c40e7b3cf2ULL, 0x8020ec2ccb1ba196ULL, 0x6693b6e09e59e313ULL,
    0x0000ce19cc7c83ebULL, 0x20cb5735f6479c3bULL, 0x762ebf3759d75a5bULL, 0x207bfe823d693975ULL,
    0xd77dc112339cd9d5ULL, 0x9ba7834284627d03ULL, 0x217dc513e95f51e9ULL, 0xb27b1a29fc5e7816ULL,
    0x00d5cd9831bb662dULL, 0x71e39b806d75734cULL, 0x7e572af006fb1a23ULL, 0xa2734f2f6ae91f85ULL,
    0xbf82c6b5022cddf2ULL, 0x5c3beac60761a0deULL, 0xcdc893bb47416998ULL, 0x6d1085615c187e01ULL,
    0x77f8ae30ac277c5dULL, 0x917c6b81122a2c91ULL, 0x5b75b699add16967ULL, 0x0000cf6ae79a069bULL,
    0xf3c40afa60de1104ULL, 0x2063127aa59167c3ULL, 0x621de62269d1894dULL, 0xd188ac1de62b4726ULL,
    0x107036e2154b673cULL, 0x0000b85f28553a1dULL, 0xf2ef4e4c18236f3dULL, 0xd9d6de6611b9f602ULL,
    0xa1fc7955fb47911cULL, 0xeb85fd032f298dbdULL, 0xbe27502fb3befae1ULL, 0xe3034251c4cd661eULL,
    0x441364d354071836ULL, 0x0082b36c75f2983eULL, 0xb145910316fa66f0ULL, 0x021c069c9847caf7ULL,
    0x2910dfc75a4b5221ULL, 0x735b353e1c57a8b5ULL, 0xce44312ce98ed96cULL, 0xbc942e4506bdfa65ULL,
    0xf05086a71257941bULL, 0xfec3b215d351ceadULL, 0x00ae1055e0144202ULL, 0xf54b40846f42e454ULL,
    0x00007fd9c8bcbcc8ULL, 0xbfbd9ef317de9bfeULL, 0xa804302ff2854e12ULL, 0x39ce4957a5e5d8d4ULL,
    0xffb9e2a45637ba84ULL, 0x55b9ad1d9ea0818bULL, 0x00008acbf319178aULL, 0x48e2bfc8d0fbfb38ULL,
    0x8be39841e848b5e8ULL, 0x0e2712160696a08bULL, 0xd51096e84b44242aULL, 0x1101ba176792e13aULL,
    0xc22e770f4531689dULL, 0x1689eff272bbc56cULL, 0x00a92a197f5650ecULL, 0xbc765990bda1784eULL,
    0xc61441e392fcb8aeULL, 0x07e13a2ced31e4a0ULL, 0x92cbe984234e9d4dULL, 0x8f4ff572bb7d8ac5ULL,
    0x0b9670c00b963bd0ULL, 0x62955a581a03eb01ULL, 0x645f83e5ea000254ULL, 0x41fce516cd88f299ULL,
    0xbbda9748da7a98cfULL, 0x0000aab2fe4845faULL, 0x19761b069bf56555ULL, 0x8b8f5e8343b6ad56ULL,
    0x3e5d1cfd144821d9ULL, 0xec5c1e2ca2b0cd8fULL, 0xfaf7e0fea7fbb57fULL, 0x000000d3ba12961bULL,
    0xda3f90178401b18eULL, 0x70ff906de33a5febULL, 0x0527d5a7c06970e7ULL, 0x22d8e773607c13e9ULL,
    0xc9ab70df643c3bacULL, 0xeda4c6dc8abe12e3ULL, 0xecef1f410033e78aULL, 0x0024c2b274ac72cbULL,
    0x06740d954fa900b4ULL, 0x1d7a299b323d6304ULL, 0xb3c37cb298cbead5ULL, 0xc986e3c76178739bULL,
    0x9fabea364b46f58aULL, 0x6da214c5af85cc56ULL, 0x17a43ed8b7a38f84ULL, 0x6eccec511d9adbebULL,
    0xf9cab30913335afbULL, 0x4a5e60c5f415eed2ULL, 0x00006967503672b4ULL, 0x9da51d121454bb87ULL,
    0x84321e13b9bbc816ULL, 0xfb3d6fb6ab2fdd8dULL, 0x60305eed8e160a8dULL, 0xcbbf4b14e9946ce8ULL,
    0x00004f63381b10c3ULL, 0x07d5b7816fcc4e10ULL, 0xe5a536726a6a8155ULL, 0x57afb23447a07fddULL,
    0x18f346f7abc9d394ULL, 0x636dc655d61ad33dULL, 0xcc8bab4939f7f3f6ULL, 0x63c7a906c1dd187bULL,
};

/* Chunk `d[0:n]`; writes each chunk's end offset to ends (cap entries at least
 * n / (target / 8) + 2) and returns the number of chunks. */
size_t xet_chunk(const uint8_t *d, size_t n, size_t target, size_t *ends, size_t cap) {
    uint64_t m = (uint64_t)(target - 1);
    uint64_t mask = m << __builtin_clzll(m);
    size_t min = target / 8, max = target * 2;
    size_t skip = (64 < min) ? min - 64 - 1 : 0;
    size_t pos = 0, k = 0;
    while (pos < n && k < cap) {
        size_t lim = pos + max < n ? pos + max : n;
        size_t i = pos + skip, cut = 0;
        uint64_t h = 0;
        for (; i < lim; i++) {
            h = (h << 1) + GEAR[d[i]];
            if ((h & mask) == 0 && i + 1 - pos >= min) {
                cut = i + 1;
                break;
            }
        }
        if (!cut)
            cut = lim;
        ends[k++] = cut;
        pos = cut;
    }
    return k;
}
