// -*- mode:objc -*-
#import <Foundation/Foundation.h>

typedef NS_ENUM(int, MjOya) {
  Oya0 = 0, Oya1, Oya2, Oya3
}; 

@interface MjLog : NSObject
  @property (readonly) NSString *seed;
  @property (readonly) NSArray <NSNumber *> *dices;
  @property (readonly) NSArray <NSArray <NSNumber *> *> *allRounds;
  @property (readonly) NSArray <NSDictionary <NSNumber *, NSNumber *> *> *deadWalls;
  @property (readonly) NSUInteger rounds;

@end

@interface NSNumber (Dice)
  - (BOOL) isEqualto: (int) d1 and: (int) d2;
@end

@interface MjLogParser : NSObject <NSXMLParserDelegate>

  @property (readonly) MjLog *mlog;
  - (void)parser:(NSXMLParser *)parser 
 didStartElement:(NSString *)elementName 
    namespaceURI:(NSString *)namespaceURI 
   qualifiedName:(NSString *)qName 
      attributes:(NSDictionary<NSString *,NSString *> *)attributeDict;
@end
