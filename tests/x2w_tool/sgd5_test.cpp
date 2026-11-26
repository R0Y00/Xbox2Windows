// Port of the upstream Rust game-format tests in src/sgd5.rs. The raw deflate streams are what
// zlib-rs makes there (level 6, window bits -15); they were made with Python's zlib
// (compressobj(6, DEFLATED, -15)), which writes the same format, and are embedded as hex.
#include <string>
#include <vector>

#include "sgd5.h"
#include "test_util.h"

using x2w::tool::sgd5::is_sgd5;
using x2w::tool::sgd5::MAGIC;
using x2w::tool::sgd5::records;

namespace {

const char* const FIRST =
    "4bcb2c2a2e5148cecf2b49cccc4b2d0200"
    ;
const char* const SEVENS =
    "63671f05c40200"
    ;
const char* const ONES =
    "6364a40c0000"
    ;
const char* const EMPTY =
    "0300"
    ;
const char* const BIG =
    "ed996357e06d18c473ad6de516b53673cbd6566e6d33d76eadeddcda666e6d33b7b6f9dcd739d7f566bec173ceff13cccc6fe6dde8d4e938"
    "7cdea6930fbe5769d97fdaaa8337de9769d8635ce4ce0bcf8dad5d7c82635372fed6693f7ccea6e30fbe5669de7fcaaafd37de96b1ed3126"
    "72fb8542634b17afe0e8d427bfeb741c3c6bd3c93b9fabb4ec3d69d5c12bafcb34ec322a726746beb1b5834770dce9277febb41e3c67d3e1"
    "3b5fab34ee3d65d5ee2b6fcbd4ed3226323ea3d0d8d4c12b84b4c903699307d2260f4a3b5b79f0256df240dae481b4c983d24e511e5c499b"
    "3c903679206df2a0b46395071bd2260fa44d1e489b3c28ed60e5c144720b07ce9dcd1c5e486ee1c0b95398c345c92d1c38772c73d825b985"
    "03e70e660e51925b38706e1fe6305e720b07ceedc21cdc25b770e0dcd6cca191e4160e9cdb98399485beb3610f1fa0ef14d8c34de83b16f6"
    "7008fa0e863dac86be7d600fd3a16f17d8c300e8db1af6d00afa36863d5485be9fc31e7e40df17600f0fa1ef9db08753d07724ec6133f43d"
    "0ef6301ffaee017b18017d37843d7482becbc01e4c3bd59e3fecd4c61ff75b559edeefd0ca0fd71b951edf7dd7e217e76d8c7c9de3827252"
    "9e76aa397fc8a9f53feeb6aa38bdcfa1e51fae362a39beebae8817993606be8e7101b97f1e77aa3d7bd0a98d5f6eb7aa3cb9d7a1956f2e37"
    "2a3dbaf3aec505e936469ef671c19bff3ced5475f69053abbfdc6d5576729f43516fae363219dd75574841a68d8ea7633c699307d2260fa4"
    "4d1e9476b2f2e042dae481b4c903699307a59da33cf8913679206df240dae4416907290fc6a44d1e489b3c90367950da71ca435dc92d1c38"
    "773273b820b98503e7ce610e2f25b770e0dc41cc2152720b07ce1dc71c764b6ee1c0b99d99430fc92d1c38b72f739820b98503e736620e65"
    "24b770e0dc36cca131f49d0c7bb8017de7c01e3e42df41b08755d0771cece130f4ed0c7be80f7dfbc21e6640df46b0872ad0b70deca135f4"
    "7d1ef6f000fa7e017bf8097d2f863d6c82be77c11e4e43dfdd610fc3a1eff1b08705d07769d8431de8bb11ecc17ee4825aed1ffddc707ce0"
    "8c4acd6f7d5cb1bfe78452b6975e2edaeee6676899961b187376e4bc5a6d1f7ddf7074e0b44a4d6fbd5fb1b7e7b852f52f3d5f94e8e66368"
    "9e961db4e5f4c805d55a3ffab9e6f0c019e51adffab86477cf09c5ea5e7a191aefe6a76b9a96f77bcbd991b3aab57df479cdd18193ca35bd"
    "f57ac9de9ea38ad5bf941f9ae8e6a16b9e4edae481b4c903699307a51dad3c58913679206df240dae44169072a0f46a44d1e489b3c903679"
    "50dab9ca833f699307d2260fa44d1e94769af2d059720b07ce1dcd1c76486ee1c0b90399c362c92d1c38772e737825b98503e74e630e9725"
    "b770e0dc96ccc14e720b07ce6dc81c4a4b6ee1c0b9fd98c344c92d1c38b71b73e8057d47c31e0e40df81b08795d0772eece113f49d067bb8"
    "0d7d5bc21e5a40df86b087cad0b71fec6126f4ed067b18047d6f873d9c80be17c11e3642df2f610fbfa0ef4bb087c7d0b72deca103f45d0a"
    "f6501bfa9e007b58087df7843d782c1cd1a1e6af8727d6cf1cd0a2e2a79b07964f74b72bf9eae28e087f572b83bcd498c0ad0b8775a8feeb"
    "fe89b533fbb528ffe9fa81a513bbdb157f757e4798bfb3955e5e72ec99cd0b47b4a9faebe191d533073429fbe9e69ea889eef54c5e5d4c08"
    "f17735d3c94b7b7c66ebc2416daaffba7d64edcc5e4dca7fbabc67e9c4cef58abf4a4f08f3b737d3cb276df240dae481b4c983d20e501e0c"
    "499b3c903679206df2a0b46394076bd2260fa44d1e489b3c28ed54e5c18db4c903699307d2260f4a3b4f791825b98503e70e600e8b24b770"
    "e0dc31cc61a7e4160e9c3b95395c92dcc28173e73187d7925b38706e03e6504a720b07ce6dc51c1a4a6ee1c0b95d99434fc92d1c38b73f73"
    "98047d07c01e5640df31b08783d0772aece116f49d077bf80c7d1bc01e2a41df56b08796d0b72bec6120f4ed0f7b98057d47c01e3640df3b"
    "600f27a1ef8bb08747d0f72bd8c36fe8bb24eca116f46d077be8087dbbc31e4642df13610f01dbce3ef95da3ede059eb8edef95ca169ef49"
    "cbf65e795da27e9751e18919f9fae60e9e0bb69d7ef2b346ebc133d61dbef3b142e3de1396edbef2b244dd2e7ee1f119b9faa68ec3e76d3b"
    "fbe07b8db6fda7ad3b7ae37d85a63dc62ddb7be17989fa2e3ee18929d9fa16ed872fd876fcc1cf1acdfbcf58b7ffc6c70ab63d262cdb7ee1"
    "65094b17bff0e8945c03d2260fa44d1e489b3c286d0fe5a180b4c903699307d2260f4adb4179c8246df240dae481b4c983d236571e92489b"
    "3c903679206df2a0b4f5958708c92d1c38b70773182db98503e776600e5d25b770e0dce6cca181e4160e9c5b9f399494dcc28173e7338737"
    "925b3870ee0ce67055720b07ce9dc81cf6496ee1c0b9c399c372e8db03f63019fa76803df481becd610fcda06f7dd84345e83b1ff6f005fa"
    "ce803ddc85be13610fc7a0ef70d8c37ae87b14ec6136f4dd05f63004faae0f7b68077d97803dd484be5fc31efe40df57600f4fa1efbdb087"
    "73d0f732d843f4b9ad7f1eb7ab3e7bd0b1b55f6e372b3fb9d7bea56f2e37283eba73525841ba859ea7c388739bff3c6c5775f68063abbfdc"
    "6c5676b2fbbea837171b988c764d0a2948b5d0f19a3fecdcd61ff7db559fdeefd8da0fd79b951fdf7ddfd217e71b14f7754e0acb49b6d0af"
    "397fc4b9f53f1eb6ab387dc0b1e51f6e362b39de7d5fc48b8b0d0c7c5d930272522d499b3c903679206df2a0b4ed95870cd2260fa44d1e48"
    "9b3c286d4fe5a190b4c903699307d2260f4a5b4f7908276df240dae481b4c983d2b6501eb64b6ee1c0b9ed994317c92d1c38b727731823b9"
    "8503e7d6630e2524b770e0dc16ccc156720b07ce9dce1cae486ee1c0b90b98c35bc92d1c387718735826b98503e74e620efba16f7bd8436f"
    "e8db13f63005fad6833d5480be2d600fcda1ef74d8c31de8bb00f6f015fa0e833dac83be93600fc7a1efceb087c1d0f768d8c31ce8bb38ec"
    "a106f4dd00f6d01efabe0c7b78027dbf813dfc85be97c21eb641dffb600fc9cffe6e393d744eb5d6f7beae39dc774ab9c6d7de2ed9dd6d4c"
    "b1ba5985a1f14e5eba661d9ffdde7272e8ac6a2def7d5e73b0efa4720dafbd5eb2b3dba862d659f9a1b14e1e7ab5da3ffbbbe1f8d039959a"
    "dffbba627fdf29a56cafbd5db4bddb1843cbacc2c06827ef79b53a3efbbee1e4d069955ade7bbfe260df71a51a5e7bbe6867371f43ebacec"
    "c05867d2260fa44d1e489b3c286d53e52181b4c903699307d2260f4a5b577908236df240dae481b4c983d2f6521e8a489b3c903679206df2"
    "a0b49d9487f3925b38706e53e6504f720b07ceadcb1c8a4b6ee1c0b9bd98c358c92d1c38b71373e82eb98503e78e670e7b24b770e0dca1cc"
    "61a9e4160e9cbb9039bc93dcc28173673187ebd0b729eca109f4ad0b7b280f7d7bc11ea642df4eb0877ed0773cece108f41d0a7b580b7d17"
    "c21ebe41df59b087fbd0775dd8431be8bb18eca13af43d06f63017faee067b18067def863d9c81be97c01eb642df6f610fffa0ef6bb087ec"
    "7f4fcf6c9e3ba44dd56f778fac9edaa749d97757f7448ded5acfa4283321c4dbd14cb7f6bfc76736ce1dd4a6f2b7db47564eedd5a4f4bbcb"
    "7b168fed5ccfa8283d21c8dbdebc43cd7f4f4fac9f3ba445c56f770f2c9fdac7aee4bbab3b22c676b53228ca8c09f0761ad6a1f6bffb2736"
    "ceedd7a2f2b7eb07564eed6e57faddf91d8bc73a5b191525c704f9903679206df240dae44169eb280fa1a44d1e489b3c90367950da66ca43"
    "22699307d2260fa44d1e94b6a3f29045dae481b4c903699307a5edad3c3c97dcc28173eb308762925b38706e33e6505f720b07ceedc81cba"
    "496ee1c0b9bd99c338c92d1c387708735822b98503e74e600e7b25b770e0dc99cce19ae4160e9cbb8839bc87be75600fe5a06f33d84353e8"
    "db11f6d017faf6863d4c83be43600f6ba0ef04d8c351e83b13f6700ffa2e823d7c87be4d600fd5a0ef7ab087b6d07757d8c350e87b2cec61"
    "1ef41d057bd8027def813d9c85beafc21e9e41dfef600f3adaffaffdffdaffaffdffdaffaffdffdaffaffdffdaffaffdffdaffaffdffdaff"
    "affdffdaffaffdffdaffaffdffdaffaffdffdaffaffdffdaffaffdffdaffaffdffdaffaffdffdaffaffdffdaffffbfffffff03"
    ;

std::vector<std::uint8_t> unhex(const char* h) {
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; h[i] && h[i + 1]; i += 2) out.push_back(std::uint8_t(std::stoi(std::string(h + i, 2), nullptr, 16)));
    return out;
}

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

std::vector<std::uint8_t> record_bytes(const std::string& name, std::uint8_t kind, std::size_t payload_len,
                                       const std::vector<std::uint8_t>& packed) {
    const auto n = std::uint32_t(packed.size() + 12);
    std::vector<std::uint8_t> out = {0xF4, 0x01, 0, 0, std::uint8_t(name.size() + 1)};
    out.insert(out.end(), name.begin(), name.end());
    out.push_back(0);
    out.push_back(kind);
    for (std::uint32_t v : {n, 0x000DDCE7u, n, std::uint32_t(payload_len)}) put32(out, v);
    out.insert(out.end(), packed.begin(), packed.end());
    return out;
}

void set32(std::vector<std::uint8_t>& b, std::size_t at, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b[at + i] = std::uint8_t(v >> (8 * i));
}

void reads_records() {
    const std::string long_name = "PS_" + std::string(40, 'x') + "_Wrapped";
    std::vector<std::uint8_t> file(MAGIC, MAGIC + 4);
    file.insert(file.end(), {1, 2, 3, 4});
    const auto first = record_bytes("VS_Blit_0011aabb_Wrapped", 0x32, 15, unhex(FIRST));
    file.insert(file.end(), first.begin(), first.end());
    // A marker whose sizes disagree is not a record.
    file.insert(file.end(), {9, 0, 0, 0, 0xE7, 0xDC, 0x0D, 0, 8, 0, 0, 0});
    const auto second = record_bytes(long_name, 0x41, 300, unhex(SEVENS));
    file.insert(file.end(), second.begin(), second.end());

    CHECK(is_sgd5(file));
    const auto found = records(file);
    CHECK(found.size() == 2);
    if (found.size() != 2) return;
    CHECK(found[0].name && *found[0].name == "VS_Blit_0011aabb_Wrapped");
    CHECK(found[0].kind == 0x32);
    const std::string text = "first container";
    CHECK(found[0].unpack() == std::vector<std::uint8_t>(text.begin(), text.end()));
    CHECK(found[1].name && *found[1].name == long_name);
    CHECK(found[1].unpack() == std::vector<std::uint8_t>(300, 7));

    // The same file inside an archive.
    std::vector<std::uint8_t> archive(0x1000, 0x5A);
    archive.insert(archive.end(), file.begin(), file.end());
    CHECK(is_sgd5(archive));
    CHECK(records(archive).size() == 2);
    CHECK(!is_sgd5(std::span<const std::uint8_t>(archive).first(0x1000)));
}

void corrupt_stream_does_not_unpack() {
    auto file = record_bytes("CS_x", 0x3c, 64, unhex(ONES));
    // (The stream itself is good.)
    CHECK(records(file).at(0).unpack() == std::vector<std::uint8_t>(64, 1));
    file[27] = 0xFF;  // first packed byte: reserved block type
    const auto r = records(file);
    CHECK(r.size() == 1);
    CHECK(!r.at(0).unpack());
}

void handles_empty_and_rejects_mismatched_or_truncated_streams() {
    const auto empty = record_bytes("CS_empty", 0x3c, 0, unhex(EMPTY));
    CHECK(records(empty).at(0).unpack() == std::vector<std::uint8_t>());

    std::vector<std::uint8_t> payload;
    for (std::uint32_t i = 0; i < 16384; ++i) payload.push_back(std::uint8_t((i * 31u) ^ (i >> 5)));
    const auto good = record_bytes("PS_big", 0x32, payload.size(), unhex(BIG));
    const auto r = records(good).at(0);
    CHECK(r.unpack() == payload);

    // Oversized stream (declared unpacked_size smaller than actual output).
    auto smaller = good;
    set32(smaller, r.offset + 8, std::uint32_t(payload.size() - 1));
    CHECK(!records(smaller).at(0).unpack());

    // Under-filled stream (declared unpacked_size larger than actual output).
    auto larger = good;
    set32(larger, r.offset + 8, std::uint32_t(payload.size() + 1));
    CHECK(!records(larger).at(0).unpack());

    // Truncated raw deflate stream.
    std::vector<std::uint8_t> truncated(good.begin(), good.end() - 1);
    const auto new_n = std::uint32_t(truncated.size() - r.offset);
    set32(truncated, r.offset - 4, new_n);
    set32(truncated, r.offset + 4, new_n);
    CHECK(!records(truncated).at(0).unpack());
}

// Additions: a name without its length byte, a stored block with bytes after the stream's end, a
// marker at the very start or end of the file, and a record whose data runs past the file.
void edges() {
    // Stored block: BFINAL 1, BTYPE 00, LEN 3, NLEN, "abc"; then trailing bytes (ignored).
    const std::vector<std::uint8_t> stored = {0x01, 3, 0, 0xFC, 0xFF, 'a', 'b', 'c', 0xAA, 0xBB};
    auto rec = record_bytes("PS_s", 0x40, 3, stored);
    CHECK(records(rec).at(0).unpack() == std::vector<std::uint8_t>({'a', 'b', 'c'}));
    CHECK(records(rec).at(0).kind == 0x40);
    // The length byte disagrees: no name, the record is still read.
    rec[4] = 9;
    CHECK(records(rec).size() == 1 && !records(rec).at(0).name);
    // No NUL before the kind byte: no name.
    rec = record_bytes("PS_s", 0x40, 3, stored);
    rec[9] = 'z';
    CHECK(records(rec).size() == 1 && !records(rec).at(0).name);
    // The marker at offset 0 (nothing before it) and data that runs past the file.
    const std::vector<std::uint8_t> head = {0xE7, 0xDC, 0x0D, 0, 20, 0, 0, 0, 1, 0, 0, 0};
    CHECK(records(head).empty());
    auto cut = record_bytes("PS_s", 0x40, 3, stored);
    cut.pop_back();
    cut.pop_back();
    cut.pop_back();
    // (n still claims the 3 bytes the file no longer has.)
    CHECK(records(cut).empty());
    // An empty name ("length" 1, just the NUL).
    rec = record_bytes("", 0x32, 3, stored);
    CHECK(records(rec).at(0).name && records(rec).at(0).name->empty());
    CHECK(!is_sgd5(std::vector<std::uint8_t>{'S', 'G', 'D'}));
}

}  // namespace

int main() {
    reads_records();
    corrupt_stream_does_not_unpack();
    handles_empty_and_rejects_mismatched_or_truncated_streams();
    edges();
    return test_util::finish("x2w_tool_sgd5_test");
}
