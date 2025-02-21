#import <Foundation/Foundation.h>
#import "mjlog.h"

@implementation MjLog
- (NSUInteger) rounds {
  return [self.allRounds count];
}
@end

@implementation MjLogCtrl {
@private
  NSString *seed;
  NSMutableArray <NSDecimalNumber *> *dices;
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
  NSString *prefix = @"mt19937ar-sha512-n288-base64,";
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

- (void) showDora: (NSNumber *)tile {
  deadWall[[NSNumber numberWithInt:dora*2 + 5]] = tile;
  dora++;
}

- (void) rinShan: (NSNumber *)tile {
  int ord;
  switch (kong) {
  case 0: ord = 1; break;
  case 1: ord = 0; break;
  case 2: ord = 3; break;
  case 3: ord = 2; break;
  default:
    NSAssert(kong > 3,@"Kong more than four times ?!");
    return;
  }
  deadWall[@(ord)] = tile;
  kong++;
}

- (void) endRound {
  [allRounds addObject:currentHand];
  [deadWalls addObject:deadWall];
  deadWall = nil;
  currentHand = nil;
}

@end

NSArray <NSNumber *> *stringToNarray(NSString *string) {
  NSArray *array = [string componentsSeparatedByString:@","];
  NSMutableArray *numberArray = [NSMutableArray arrayWithCapacity:13];
  for (NSString *numberString in array) {
    NSNumber *number = @([numberString integerValue]);
    [numberArray addObject:number];
  }
  return numberArray;
}

BOOL isFetchTileAction(NSString *string, int *number) {
  if (!([string hasPrefix:@"T"] ||
        [string hasPrefix:@"U"] ||
        [string hasPrefix:@"V"] ||
        [string hasPrefix:@"W"]))
    return NO;
  NSScanner *sc = [NSScanner scannerWithString:[string substringFromIndex:1]];
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

- (void)parserDidStartDocument:(NSXMLParser *)parser {
  mlog = [MjLogCtrl alloc];
}

- (void)parser:(NSXMLParser *)parser 
didStartElement:(NSString *)elementName 
  namespaceURI:(NSString *)namespaceURI 
 qualifiedName:(NSString *)qName 
    attributes:(NSDictionary<NSString *,NSString *> *)attributeDict {
  int num;
  @autoreleasepool {
    if (isFetchTileAction(elementName, &num)) {
      // NSLog(@"draw tile");
      if (kong == NO) {
        [mlog draw:@(num)];
      } else {
        [mlog rinShan:@(num)];
        kong = NO;
      }
    } else if ([elementName isEqualToString:@"INIT"]) {
      NSArray <NSNumber *> *seed = stringToNarray([attributeDict objectForKey:@"seed"]);
      [mlog.dices addObject:@(seed[3].intValue*10+seed[4].intValue)];
      kong = NO;
      NSInteger oya = [[attributeDict objectForKey:@"oya"] integerValue];
      [mlog startHand:oya player0:stringToNarray([attributeDict objectForKey:@"hai0"])
              player1:stringToNarray([attributeDict objectForKey:@"hai1"])
              player2:stringToNarray([attributeDict objectForKey:@"hai2"])
              player3:stringToNarray([attributeDict objectForKey:@"hai3"])];
      [mlog showDora:seed[5]];
    } else if ([elementName isEqualToString:@"AGARI"] ||
               [elementName isEqualToString:@"RYUUKYOKU"]) {
      [mlog endRound];
    } else if ([elementName isEqualToString:@"SHUFFLE"]) {
      mlog = [mlog initWithSeed:[attributeDict objectForKey:@"seed"]];
    } else if ([elementName isEqualToString:@"DORA"]) {
      kong = YES;
      [mlog showDora:@([[attributeDict objectForKey:@"hai"] integerValue])];
    } else if ([elementName isEqualToString:@"mjloggm"]) {
      if (![[attributeDict objectForKey:@"ver"] isEqualToString:@"2.3"])
        NSLog(@"Log format version changed!");
    }
  }
}

@end
