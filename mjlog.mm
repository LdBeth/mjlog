// -*- mode:objc -*-
#import <Foundation/Foundation.h>
#import "mjlog.h"
#include <string>

@implementation MjLog
- (NSUInteger) rounds {
  return [self.allRounds count];
}
@end

@interface MjLogCtrl : MjLog
  @property (retain, readwrite) NSMutableArray *dices;
  - (instancetype) initWithSeed:(NSString*)seedString;
  - (void) startHand:(enum MjOya)oya
             player0:(NSArray *)hand0
             player1:(NSArray *)hand1
             player2:(NSArray *)hand2
             player3:(NSArray *)hand3;
  - (void) roll:(NSNumber *)d1 and:(NSNumber *) d2;
  - (void) draw:(NSNumber *)tile;
  - (void) showDora:(NSNumber *)tile;
  - (void) rinShan:(NSNumber *)tile;
  - (void) endRound;
@end

@implementation MjLogCtrl {
@private
  NSString *seed;
  NSMutableArray <NSNumber *> *dices;
  NSMutableArray *allRounds;
  NSMutableArray *deadWalls;
  NSMutableArray *currentHand;
  NSMutableDictionary <NSNumber *, NSNumber *> *deadWall;
  NSUInteger dora, kong;
}

@synthesize seed, dices, allRounds, deadWalls;

- (instancetype) init {
  self = [super init];
  kong = dora = 0;
  return self;
}

- (instancetype) initWithSeed:(NSString*)seedString {
  self = [self init];
  auto prefix = @"mt19937ar-sha512-n288-base64,";
  NSAssert([seedString hasPrefix:prefix], @"seed format incorrect!");
  seed = [seedString substringFromIndex:[prefix length]];
  allRounds = [NSMutableArray arrayWithCapacity:20];
  dices = [NSMutableArray arrayWithCapacity:20];
  return self;
}

- (void) startHand:(enum MjOya)oya
           player0:(NSArray *)hand0
           player1:(NSArray *)hand1
           player2:(NSArray *)hand2
           player3:(NSArray *)hand3 {
  NSArray *h[4];
  switch (oya) {
  case Oya0:
    h[0] = hand0; h[1] = hand1; h[2] = hand2; h[3] = hand3;
    break;
  case Oya1:
    h[3] = hand0; h[0] = hand1; h[1] = hand2; h[2] = hand3;
    break;
  case Oya2:
    h[2] = hand0; h[3] = hand1; h[0] = hand2; h[1] = hand3;
    break;
  case Oya3:
    h[1] = hand0; h[2] = hand1; h[3] = hand2; h[0] = hand3; 
    break;
  }
  currentHand = [NSMutableArray arrayWithCapacity:136];
  deadWall = [[NSMutableDictionary alloc] initWithCapacity:14];
  for (int i = 0; i<4; ++i) {
    NSAssert([h[i] count] == 13, @"Invalid hand! %@", h[i]);
  }

  for (int i=0,s=0; i < 3; ++i,s=i*4) {
    for(int j=0;j<4;++j){
    [currentHand addObjectsFromArray:
                   @[h[j][s],h[j][s+1],h[j][s+2],h[j][s+3]]];
    }
  }
  [currentHand addObjectsFromArray:
                 @[h[0][12],h[1][12],h[2][12],h[3][12]]];

}

- (void) draw: (NSNumber *)tile {
  [currentHand addObject: tile];
}

- (void) roll:(NSNumber *)d1 and:(NSNumber *)d2 {
  [dices addObject:@((d1.intValue<<3)+d2.intValue)];
}

- (void) showDora: (NSNumber *)tile {
  deadWall[[NSNumber numberWithInt:dora*2 + 5]] = tile;
  dora++;
}

- (void) rinShan: (NSNumber *)tile {
  int ord[] = {1, 0, 3, 2};
  NSAssert(kong <= 3,@"Kong more than four times ?!");
  deadWall[@(ord[kong])] = tile;
  kong++;
}

- (void) endRound {
  [allRounds addObject:currentHand];
  [deadWalls addObject:deadWall];
  deadWall = nil;
  currentHand = nil;
}

@end

@implementation NSNumber (Dice)
- (BOOL)isEqualto:(int)d1 and:(int)d2 {
  return self.intValue == (d1<<3)+d2;
}
@end

NSArray <NSNumber *> *stringToNarray(NSString *string) {
  auto array = [string componentsSeparatedByString:@","];
  auto numberArray = [NSMutableArray arrayWithCapacity:13];
  for (NSString *numberString in array) {
    [numberArray addObject:@([numberString integerValue])];
  }
  return numberArray;
}

BOOL isFetchTileAction(std::string string, int *number) {
  if (string[0] != 'T' &&
      string[0] != 'U' &&
      string[0] != 'V' &&
      string[0] != 'W')
    return NO;
  NSScanner *sc = [NSScanner scannerWithString:
                               @(string.substr(1).c_str())];
  [sc scanInt:number];
  BOOL isDecimal = [sc isAtEnd];
  return isDecimal;
}

@implementation MjLogParser {
@private
  MjLogCtrl *mlog;
  BOOL kong;
}
@synthesize mlog;

- (void)parser:(NSXMLParser *)parser 
didStartElement:(NSString *)elementName 
  namespaceURI:(NSString *)namespaceURI 
 qualifiedName:(NSString *)qName 
    attributes:(NSDictionary<NSString *,NSString *> *)attributeDict {
  int num;
  @autoreleasepool {
    auto name = std::string([elementName UTF8String]);
    if (isFetchTileAction(name, &num)) {
      // NSLog(@"draw tile");
      if (kong == NO) {
        [mlog draw:@(num)];
      } else {
        [mlog rinShan:@(num)];
        kong = NO;
      }
    } else if (name == "INIT") {
      auto seed = stringToNarray(attributeDict[@"seed"]);
      [mlog roll: seed[3] and: seed[4]];
      kong = NO;
      auto oya = static_cast<MjOya>([attributeDict[@"oya"] intValue]);
      [mlog startHand:oya
              player0:stringToNarray(attributeDict[@"hai0"])
              player1:stringToNarray(attributeDict[@"hai1"])
              player2:stringToNarray(attributeDict[@"hai2"])
              player3:stringToNarray(attributeDict[@"hai3"])];
      [mlog showDora:seed[5]];
    } else if (name == "AGARI" ||
               name == "RYUUKYOKU") {
      [mlog endRound];
    } else if (name == "SHUFFLE") {
      mlog = [[MjLogCtrl alloc] initWithSeed:attributeDict[@"seed"]];
    } else if (name == "DORA") {
      kong = YES;
      [mlog showDora:@([attributeDict[@"hai"] integerValue])];
    } else if (name == "mjloggm") {
      if (![attributeDict [@"ver"] isEqualToString:@"2.3"])
        NSLog(@"Log format version changed!");
    }
  }
}

@end
