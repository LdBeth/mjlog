// -*- mode:objc -*-
#import <Foundation/Foundation.h>

#ifdef __cplusplus
extern "C" {
#endif

enum MjOya {
  Oya0, Oya1, Oya2, Oya3
}; 

@interface MjLog : NSObject {
  NSString *seed;
  NSMutableArray <NSDecimalNumber *> *dices;
  NSMutableArray *allRounds;
  NSMutableArray *currentHand;
}

  @property (nonatomic, retain) NSString *seed;
  @property (nonatomic, retain) NSMutableArray <NSDecimalNumber *> *dices;
  @property (nonatomic, retain) NSMutableArray *allRounds;
  @property (nonatomic, retain) NSMutableArray *currentHand;

  - (MjLog *) initWithSeed:(NSString*)seedString;
  - (void) startHand:(enum MjOya)oya
             player0:(NSArray *)hand0
             player1:(NSArray *)hand1
             player2:(NSArray *)hand2
             player3:(NSArray *)hand3;
  - (void) draw:(NSNumber *)tile;
  - (void) endRound;
  - (NSUInteger) rounds;
@end

@interface MjLogParser : NSObject <NSXMLParserDelegate> {
  MjLog *mlog;
  BOOL kong;
}
  @property (nonatomic, retain) MjLog *mlog;
  @property BOOL kong;

  - (MjLogParser*) initXMLParser;
  - (void)parser:(NSXMLParser *)parser 
 didStartElement:(NSString *)elementName 
    namespaceURI:(NSString *)namespaceURI 
   qualifiedName:(NSString *)qName 
      attributes:(NSDictionary<NSString *,NSString *> *)attributeDict;
@end

#ifdef __cplusplus
}
#endif
