// -*- mode:c++ -*-
#import "mjlog.h"
#import "mt19937ar.h"
#import <CommonCrypto/CommonDigest.h>
#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <zlib.h>

#define SHA512_DIGEST_SIZE CC_SHA512_DIGEST_LENGTH

static const char *haiDisp[] = {
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "①",  "②",  "③",
    "④",  "⑤",  "⑥",  "⑦",  "⑧",  "⑨",  "１", "２", "３", "４", "５", "６",
    "７", "８", "９", "東", "南", "西", "北", "白", "發", "中"};

static bool verbose = false;

template <typename T, size_t N> void printBlock(const T (&arr)[N]) {
  for (size_t i = 0; i < N; ++i) {
    printf(" %08X", arr[i]);
    if ((i + 1) % 8 == 0)
      std::cout << std::endl;
  }
}

void setup_seed(_MTRND &mt, char *bytes, NSString *data) {
  uint32_t seed[624];

  auto *d = [[NSData alloc]
      initWithBase64EncodedString:data
                          options:NSDataBase64DecodingIgnoreUnknownCharacters];
  [d getBytes:seed length:sizeof(seed)];

  mt.init_by_array(seed, sizeof(seed) / sizeof(*seed));
  for (int i = 0; i < sizeof(seed) / sizeof(*seed); ++i) {
    snprintf(bytes + (i * 8), 9, "%08x", CFSwapInt32(seed[i]));
    // printf("%08x,", CFSwapInt32(seed[i]));
  }
  if (verbose) {
    std::cout << "mt.seed=" << std::endl;
    printBlock(seed);
    std::cout << std::endl;
  }
}

int checkMlogRounds(_MTRND &mt, MjLog *mlog) {
  int i;

  auto round = mlog.rounds;
  for (int nKyoku = 0; nKyoku < round; ++nKyoku)
    @autoreleasepool {
      uint32_t
          rnd[SHA512_DIGEST_SIZE / sizeof(uint32_t) * 9]; // 135+2以上を確保
      {
        uint32_t
            src[sizeof(rnd) / sizeof(*rnd) * 2]; // 1024bit単位で512bitへhash
        for (i = 0; i < sizeof(src) / sizeof(*src); ++i)
          src[i] = mt.genrand_int32();
        if (verbose) {
          std::cout << "src=" << std::endl;
          printBlock(src);
          std::cout << std::endl;
        }
        for (i = 0; i < sizeof(rnd) / SHA512_DIGEST_SIZE; ++i) {
          CC_SHA512((unsigned char *)src + i * SHA512_DIGEST_SIZE * 2,
                    SHA512_DIGEST_SIZE * 2,
                    (unsigned char *)rnd + i * SHA512_DIGEST_SIZE);
        }
      }
      if (verbose) {
        std::cout << "rnd=" << std::endl;
        printBlock(rnd);
        std::cout << std::endl;
      }

      unsigned char yama_arr[136]; // サンマは108
      unsigned char *yama = yama_arr;
      __block bool error = false;
      for (i = 0; i < 136; ++i)
        yama[i] = i;
      for (i = 0; i < 136 - 1; ++i)
        std::swap(yama[i],
                  yama[i + (rnd[i] % (136 - i))]); // 1/2^32以下の誤差は許容

      std::cout << "nKyoku=" << nKyoku << " yama=" << std::endl;
      for (i = 0; i < 136; ++i)
        printf("%s", haiDisp[yama[i] / 4]);
      std::cout << std::endl;
      int dice1 = rnd[135] % 6;
      int dice2 = rnd[136] % 6;
      // rnd[137]～rnd[143]は未使用
      if (verbose) {
        for (i = 0; i < 136; ++i)
          printf("%d,", yama[i]);
        std::cout << std::endl;
      }
      std::cout << "dice0=" << dice1 << " dice1=" << dice2 << std::endl;
      error = false;
      if (![mlog.dices[nKyoku] isEqualto:dice1 and:dice2]) {
        error = true;
        NSLog(@"dice mismatch!");
      }
      auto hand = mlog.allRounds[nKyoku];
      auto wall = mlog.deadWalls[nKyoku];

      // Count kans: rinshan tiles occupy dead wall positions 0-3
      NSUInteger kangCount = 0;
      for (int ki = 0; ki < 4; ki++)
        if (wall[@(ki)] != nil)
          kangCount++;
      // After kangCount kans, live wall shrinks by kangCount from the bottom
      // (replenishment consumes yama[14..14+kangCount-1]).
      // Total drawable tiles = 122 - kangCount; hand must not exceed this.
      if (hand.count > 122 - kangCount) {
        NSLog(@"Round %d: %lu draws exceed live wall capacity (%lu with %lu "
              @"kans)",
              nKyoku, hand.count, 122 - kangCount, kangCount);
        error = true;
      }

      [hand enumerateObjectsUsingBlock:^(NSNumber *n, NSUInteger idx,
                                         BOOL *stop) {
        if (n.intValue != yama[135 - idx]) {
          NSLog(@"Mismatched element at index %lu is %@", idx, n);
          error = true;
          *stop = YES;
        }
      }];
      [wall enumerateKeysAndObjectsUsingBlock:^(NSNumber *i, NSNumber *n,
                                                BOOL *stop) {
        if (n.intValue != yama[i.unsignedIntValue]) {
          NSLog(@"Mismatched deadtile at index %@ is %@", i, n);
          error = true;
          *stop = YES;
        }
      }];

      // Cross-check AGARI doraHai against dead wall tiles recorded during play.
      // doraLists[nKyoku] is empty for ryuukyoku rounds (no AGARI element).
      auto doraList = mlog.doraLists[nKyoku];
      for (NSUInteger di = 0; di < doraList.count; di++) {
        NSNumber *expected = wall[@(5 + di * 2)];
        if (![doraList[di] isEqualToNumber:expected]) {
          NSLog(@"Dora mismatch at position %lu: AGARI doraHai says %@, dead "
                @"wall has %@",
                di, doraList[di], expected);
          error = true;
        }
      }

      if (!error)
        std::cout << "Round passes check." << std::endl;
      else
        return 1;
    }
  return 0;
}

static NSData *decompressIfGzip(NSData *data) {
  auto bytes = (const uint8_t *)data.bytes;
  if (data.length < 2 || bytes[0] != 0x1f || bytes[1] != 0x8b)
    return data;

  z_stream stream = {};
  if (inflateInit2(&stream, 15 + 16) != Z_OK)
    return nil;

  auto result = [NSMutableData dataWithLength:data.length * 4];
  stream.next_in = (Bytef *)data.bytes;
  stream.avail_in = (uInt)data.length;

  int status;
  do {
    if (stream.total_out >= result.length)
      result.length *= 2;
    stream.next_out = (Bytef *)result.mutableBytes + stream.total_out;
    stream.avail_out = (uInt)(result.length - stream.total_out);
    status = inflate(&stream, Z_NO_FLUSH);
  } while (status == Z_OK);

  inflateEnd(&stream);
  if (status != Z_STREAM_END)
    return nil;

  result.length = stream.total_out;
  return result;
}

int main(int argc, char *const argv[]) {
  const char *seat = NULL;
  bool hash = false;
  int opt;
  while ((opt = getopt(argc, argv, "vhs:")) != EOF) {
    switch (opt) {
    case 'v':
      verbose = true;
      break;
    case 's':
      seat = optarg;
      [[fallthrough]];
    case 'h':
      hash = true;
      break;
    default:
      std::cerr << argv[0] << ": [-v] [-h] [-s value] mjlog" << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (argc - optind < 1) {
    std::cerr << "No input!" << std::endl;
    return -1;
  }
  const char *file = argv[optind];
  @autoreleasepool {
    BOOL success;
    MjLog *mlog;
    {
      auto url = [[NSURL alloc] initFileURLWithPath:@(file)];
      auto rawData = [[NSData alloc] initWithContentsOfURL:url];
      if (rawData == nil) {
        std::cerr << "Could not read file: " << file << std::endl;
        return -1;
      }
      auto xmlData = decompressIfGzip(rawData);
      if (xmlData == nil) {
        std::cerr << "Decompression failed" << std::endl;
        return -1;
      }
      auto xmlparser = [[NSXMLParser alloc] initWithData:xmlData];
      auto parser = [[MjLogParser alloc] init];
      [xmlparser setDelegate:parser];

      success = [xmlparser parse];
      mlog = parser.mlog;
    }
    if ((!success) || (mlog.seed == nil)) {
      std::cerr << "Parse not success" << std::endl;
      return -1;
    }

    std::cout << "East : " << [mlog.pE UTF8String] << std::endl;
    std::cout << "South: " << [mlog.pS UTF8String] << std::endl;
    std::cout << "West : " << [mlog.pW UTF8String] << std::endl;
    std::cout << "North: " << [mlog.pN UTF8String] << std::endl;

    char source[5000];
    _MTRND mt;

    setup_seed(mt, source + 4, mlog.seed);
    if (hash) {
      printf("shasum:\n");
      unsigned char checksum[SHA512_DIGEST_SIZE];
      auto printHash = [&]() {
        for (int j = 0; j < SHA512_DIGEST_SIZE; ++j)
          printf("%02x", checksum[j]);
        printf("\n");
      };
      if (seat != NULL) {
        strncpy(source, seat, 4);
        CC_SHA512(source, 8 * 624 + 4, checksum);
        printHash();
      } else {
        char p[] = "0123";
        bool canInfer = [mlog computSeat:p];
        if (!canInfer) {
          printf("[Cannot infer seat.]\n");
        }
        do {
          strncpy(source, p, 4);
          printf("(%s) ", p);
          CC_SHA512(source, 8 * 624 + 4, checksum);
          printHash();
        } while (!canInfer && std::next_permutation(p, p + 4));
      }
    }
    return checkMlogRounds(mt, mlog);
  }
}
