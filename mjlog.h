// -*- mode:objc -*-
#import <Foundation/Foundation.h>

#ifdef __cplusplus
extern "C" {
#endif

enum MjOya {
  Oya0, Oya1, Oya2, Oya3
}; 

@interface MjLog : NSObject {
@protected
  NSString *seed;
  NSMutableArray <NSDecimalNumber *> *dices;
  NSMutableArray *allRounds;
  NSUInteger rounds;
}

  @property (nonatomic, retain, readonly) NSString *seed;
  @property (nonatomic, retain, readonly) NSArray <NSDecimalNumber *> *dices;
  @property (nonatomic, retain, readonly) NSArray *allRounds;
  @property (readonly) NSUInteger rounds;

@end

@interface MjLogCtrl : MjLog {
  NSMutableArray *currentHand;
}
  @property (nonatomic, retain, readwrite) NSString *seed;
  @property (nonatomic, retain, readwrite) NSMutableArray <NSDecimalNumber *> *dices;
  @property (nonatomic, retain, readwrite) NSMutableArray *allRounds;
  @property (readwrite) NSUInteger rounds;
  @property (nonatomic, retain, readwrite) NSMutableArray *currentHand;
  - (MjLog *) initWithSeed:(NSString*)seedString;
  - (void) startHand:(enum MjOya)oya
             player0:(NSArray *)hand0
             player1:(NSArray *)hand1
             player2:(NSArray *)hand2
             player3:(NSArray *)hand3;
  - (void) draw:(NSNumber *)tile;
  - (void) endRound;
@end

@interface MjLogParser : NSObject <NSXMLParserDelegate> {
@private
  MjLogCtrl *mlog;
  BOOL kong;
}
  @property (nonatomic, retain) MjLog *mlog;
  @property BOOL kong;

  - (void)parserDidStartDocument:(NSXMLParser *)parser;
  - (void)parser:(NSXMLParser *)parser 
 didStartElement:(NSString *)elementName 
    namespaceURI:(NSString *)namespaceURI 
   qualifiedName:(NSString *)qName 
      attributes:(NSDictionary<NSString *,NSString *> *)attributeDict;
  - (MjLog *)getLog;
@end

#ifdef __cplusplus
}
#endif
