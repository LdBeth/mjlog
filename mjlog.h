// -*- mode:objc -*-
#import <Foundation/Foundation.h>

#ifdef __cplusplus
extern "C" {
#endif

enum MjOya {
  Oya0 = 0, Oya1, Oya2, Oya3
}; 

@interface MjLog : NSObject
  @property (readonly) NSString *seed;
  @property (readonly) NSArray <NSNumber *> *dices;
  @property (readonly) NSArray <NSArray *> *allRounds;
  @property (readonly) NSArray <NSDictionary <NSNumber *, NSNumber *> *> *deadWalls;
  @property (readonly) NSUInteger rounds;

@end

@interface MjLogCtrl : MjLog
  @property (retain, readwrite) NSMutableArray *dices;
  - (instancetype) initWithSeed:(NSString*)seedString;
  - (void) startHand:(enum MjOya)oya
             player0:(NSArray *)hand0
             player1:(NSArray *)hand1
             player2:(NSArray *)hand2
             player3:(NSArray *)hand3;
  - (void) draw:(NSNumber *)tile;
  - (void) showDora:(NSNumber *)tile;
  - (void) rinShan:(NSNumber *)tile;
  - (void) endRound;
@end

@interface MjLogParser : NSObject <NSXMLParserDelegate>

  @property (readonly) MjLog *mlog;
  - (void)parserDidStartDocument:(NSXMLParser *)parser;
  - (void)parser:(NSXMLParser *)parser 
 didStartElement:(NSString *)elementName 
    namespaceURI:(NSString *)namespaceURI 
   qualifiedName:(NSString *)qName 
      attributes:(NSDictionary<NSString *,NSString *> *)attributeDict;
@end

#ifdef __cplusplus
}
#endif
