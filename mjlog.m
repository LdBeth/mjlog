#include "mjlog.h"
#include <Foundation/Foundation.h>

@implementation MjLog
@synthesize seed, dices, allRounds, currentHand;

- (MjLog *) initWithSeed:(NSString*)seedString {
  NSString *prefix = @"mt19937ar-sha512-n288-base64,";
  if ([seedString hasPrefix:prefix]) {
    self.seed = [seedString substringFromIndex:[prefix length]];
  } else {
    NSLog(@"seed format incorrect!");
  }
  self.allRounds = [NSMutableArray arrayWithCapacity:20];
  self.dices = [NSMutableArray arrayWithCapacity:20];
  return self;
}

- (void) startHand:(enum MjOya)oya
           player0:(NSArray *)hand0
           player1:(NSArray *)hand1
           player2:(NSArray *)hand2
           player3:(NSArray *)hand3 {
  NSArray *h1, *h2, *h3, *h4;
  switch (oya) {
  case Oya0:
    h1 = hand0; h2 = hand1; h3 = hand2; h4 = hand3;
    break;
  case Oya1:
    h1 = hand1; h2 = hand2; h3 = hand3; h4 = hand0;
    break;
  case Oya2:
    h1 = hand2; h2 = hand3; h3 = hand0; h4 = hand1;
    break;
  case Oya3:
    h1 = hand3; h2 = hand0; h3 = hand1; h4 = hand2;
    break;
  }
  currentHand = [NSMutableArray arrayWithCapacity:136];
  if ([h1 count] != 13 || [h2 count] != 13 || [h3 count] != 13 || [h4 count] != 13) {
    NSLog(@"Invalid hand!");
  }

  for (int i=0,s=0; i < 3; ++i,s=i*4) {
    [currentHand addObjectsFromArray:@[h1[s],h1[s+1],h1[s+2],h1[s+3]]];
    [currentHand addObjectsFromArray:@[h2[s],h2[s+1],h2[s+2],h2[s+3]]];
    [currentHand addObjectsFromArray:@[h3[s],h3[s+1],h3[s+2],h3[s+3]]];
    [currentHand addObjectsFromArray:@[h4[s],h4[s+1],h4[s+2],h4[s+3]]];
  }
  [currentHand addObjectsFromArray:@[h1[12],h2[12],h3[12],h4[12]]];

}

- (void) draw: (NSNumber *)tile {
  [currentHand addObject: tile];
}

- (void) endRound {
  [allRounds addObject:currentHand];
}

- (NSUInteger) rounds {
  return [allRounds count];
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

//       局，本，？，d1，d2，dora
// seed="4,0,0,2,4,69"

@implementation MjLogParser

@synthesize mlog, kong;

- (MjLogParser *) initXMLParser
{
    self = [super init];

    mlog = [MjLog alloc];

    return self;
}

- (void)parser:(NSXMLParser *)parser 
didStartElement:(NSString *)elementName 
  namespaceURI:(NSString *)namespaceURI 
 qualifiedName:(NSString *)qName 
    attributes:(NSDictionary<NSString *,NSString *> *)attributeDict {
  int num;
  if (isFetchTileAction(elementName, &num)) {
    // NSLog(@"draw tile");
    if (kong == NO) {
      [mlog draw:[[NSNumber alloc] initWithInt:num]];
    } else kong = NO;
  } else if ([elementName isEqualToString:@"INIT"]) {
    NSArray <NSNumber *> *seed = stringToNarray([attributeDict objectForKey:@"seed"]);
    [mlog.dices addObject:[[NSDecimalNumber alloc]
                            initWithInt:seed[3].intValue*10+seed[4].intValue]];
    kong = NO;
    NSInteger oya = [[attributeDict objectForKey:@"oya"] integerValue];
    [mlog startHand:oya player0:stringToNarray([attributeDict objectForKey:@"hai0"])
            player1:stringToNarray([attributeDict objectForKey:@"hai1"])
            player2:stringToNarray([attributeDict objectForKey:@"hai2"])
            player3:stringToNarray([attributeDict objectForKey:@"hai3"])];
  } else if ([elementName isEqualToString:@"AGARI"] ||
             [elementName isEqualToString:@"RYUUKYOKU"]) {
    [mlog endRound];
  } else if ([elementName isEqualToString:@"SHUFFLE"]) {
    mlog = [mlog initWithSeed:[attributeDict objectForKey:@"seed"]];
  } else if ([elementName isEqualToString:@"DORA"]) {
    // todo: verift dora
    kong = YES;
  }
  
}

@end
