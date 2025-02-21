// -*- mode:c++ -*-
#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#import <CoreFoundation/CoreFoundation.h>
#include <algorithm>
#import "mt19937ar.h"
#import "mjlog.h"

#define SHA512_DIGEST_SIZE CC_SHA512_DIGEST_LENGTH

static const char *haiDisp[34]={
  "一","二","三","四","五","六","七","八","九",
  "①","②","③","④","⑤","⑥","⑦","⑧","⑨",
  "１","２","３","４","５","６","７","８","９",
  "東","南","西","北","白","發","中"
};

static bool verbose = false;

#define PRINT_BLOCK(var) \
  for(int i=0;i<sizeof(var)/sizeof(*var);++i) { \
    printf(" %08X",var[i]);                     \
    if (7 == i % 8)                             \
      printf("\n");                             \
  }

void setup_seed(_MTRND &mt, char *bytes, NSString *data) {
  uint32_t seed[624];
        
  NSData *d = [[NSData alloc] initWithBase64EncodedString:data
               options:NSDataBase64DecodingIgnoreUnknownCharacters];
  [d getBytes:seed length:sizeof(seed)];

  mt.init_by_array(seed,sizeof(seed)/sizeof(*seed));
  for(int i=0;i<sizeof(seed)/sizeof(*seed);++i) {
    snprintf(bytes + (i * 8), 9, "%08x", CFSwapInt32(seed[i]));
    // printf("%08x,", CFSwapInt32(seed[i]));
  }
  if (verbose) {
    printf("mt.seed=\n");
    PRINT_BLOCK(seed);
    printf("\n");
  }
}

int checkMlogRounds(_MTRND &mt, MjLog *mlog){
    int i;
    int nKyoku=0;
    NSUInteger round = mlog.rounds;
    for(;nKyoku<round;++nKyoku) @autoreleasepool {
      uint32_t rnd[SHA512_DIGEST_SIZE/sizeof(uint32_t)*9]; // 135+2以上を確保
      {
        uint32_t src[sizeof(rnd)/sizeof(*rnd)*2]; // 1024bit単位で512bitへhash
        for(i=0;i<sizeof(src)/sizeof(*src);++i) src[i]=mt.genrand_int32();
        if (verbose) {
          printf("src=\n");
          PRINT_BLOCK(src);
          printf("\n");
        }
        for(i=0;i<sizeof(rnd)/SHA512_DIGEST_SIZE;++i){
          CC_SHA512((unsigned char *)src + i * SHA512_DIGEST_SIZE * 2,
                    SHA512_DIGEST_SIZE * 2,
                    (unsigned char *)rnd + i * SHA512_DIGEST_SIZE);
        }
      }
      if (verbose) {
        printf("rnd=\n");
        PRINT_BLOCK(rnd);
        printf("\n");
      }

      static unsigned char yama[136];// サンマは108
      static bool error;
      for(i=0;i<136;++i) yama[i]=i;
      for(i=0;i<136-1;++i) std::swap(yama[i],yama[i + (rnd[i]%(136-i))]); // 1/2^32以下の誤差は許容

      printf("nKyoku=%d yama=",nKyoku);
      for(i=0;i<136;++i) printf("%s",haiDisp[yama[i]/4]);
      printf("\n");
      int dice1=rnd[135]%6;
      int dice2=rnd[136]%6;
      // rnd[137]～rnd[143]は未使用
      printf("dice0=%d dice1=%d\n",dice1, dice2);
      if (verbose) {
        for(i=0;i<136;++i) printf("%d,",yama[i]);
        printf("\n");
      }
      error = false;
      if (mlog.dices[nKyoku].intValue != dice1*10+dice2) {
        error = true;
        NSLog(@"dice mismatch!");
      }
      NSArray <NSNumber *> *hand = mlog.allRounds[nKyoku];
      [hand enumerateObjectsUsingBlock:^(NSNumber *n, NSUInteger idx, BOOL *stop) {
          if (n.intValue != yama[135-idx]) {
            NSLog(@"Mismatched element at index %lu is %@", (unsigned long)idx, n);
            error = true;
            *stop = YES;
          }
        }];
      NSDictionary <NSNumber *, NSNumber *> *wall = mlog.deadWalls[nKyoku];
      [wall enumerateKeysAndObjectsUsingBlock:
       ^(NSNumber *i, NSNumber *n, BOOL *stop){
          if (n.intValue != yama[i.unsignedIntValue]) {
            NSLog(@"Mismatched deadwall at index %lu is %@", (unsigned long)i, n);
            error = true;
            *stop = YES;
          }
        }];
      if (!error) {
        printf("Hand passes check.\n");
      } else {
        return 1;
      }
      // for(i=0;i<136;++i) printf("%d,",yama[i]);
    }
    return 0;
}

static const char *perm[] = {
"0123", "0132", "0213", "0231", "0312", "0321",
"1023", "1032", "1203", "1230", "1302", "1320",
"2013", "2031", "2103", "2130", "2301", "2310",
"3012", "3021", "3102", "3120", "3201", "3210",
};

int main(int argc, char *argv[]) {
  char *file = NULL;
  bool hash = false;
  for (int i = 1; i < argc; i++) {
    if (0 == strcmp(argv[i], "-v")) verbose = true;
    if (0 == strcmp(argv[i], "-h")) hash = true;
    else file = argv[i];
  }
  if (file == NULL) {
    NSLog(@"No input!");
    return -1;
  }
  @autoreleasepool {
    NSURL *url = [[NSURL alloc] initFileURLWithPath:[[NSString alloc] initWithUTF8String:file]];
    NSXMLParser *xmlparser = [[NSXMLParser alloc] initWithContentsOfURL:url];
    MjLogParser *parser = [MjLogParser alloc];
    [xmlparser setDelegate:parser];

    BOOL success = [xmlparser parse];
    MjLog *mlog = parser.mlog;

    if ((!success) || (mlog.seed == nil)) {
      NSLog(@"Parse not success");
      return -1;
    }
    NSString *data = mlog.seed;
    char source[5000];
    _MTRND mt;
    
    setup_seed(mt, source + 4, data);
    if (hash) {
      printf("shasum:\n");
      for (int i=0;i<24;++i) {
        strncpy(source, perm[i], 4);
        printf("(%s) ", perm[i]);
        unsigned char checksum[SHA512_DIGEST_SIZE];
        CC_SHA512(source, 8*624+4, checksum);
      
        for(int i = 0; i < SHA512_DIGEST_SIZE; ++i) {
          printf("%02x", checksum[i]);
        }
        printf("\n");
      }
    }
    return checkMlogRounds(mt, mlog);
  }
}
