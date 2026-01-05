#import <Foundation/Foundation.h>

#if TARGET_OS_SIMULATOR
// Stubbing missing Metal symbols in Simulator SDK for arm64
// These symbols are present in the device SDK but missing in some simulator SDK versions
NSString * const MTLIOErrorDomain = @"MTLIOErrorDomain";
NSString * const MTLTensorDomain = @"MTLTensorDomain";
#endif
