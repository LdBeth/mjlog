# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A macOS tool to verify the integrity of Tenhou online mahjong game logs against their cryptographic seed. It confirms that the tile draws in a replay match what the PRNG would generate from the recorded seed, proving the game was not manipulated.

**References:**
- https://tenhou.net/stat/rand/
- https://81100118.github.io/2021/01/01/天凤牌山生成算法及其验证/

## Build

```sh
sh build.sh
```

This compiles with `clang++` using C++17, Objective-C ARC, and `-O3 -flto`. The output binary is `./out`.

## Usage

```sh
./out [-v] [-h] [-s <seat>] <mjlog.xml>
```

- `-v`: verbose (prints PRNG state, raw tile indices)
- `-h`: compute SHA-512 hashes of seed for all 24 seat permutations
- `-s <seat>`: compute SHA-512 hash for a specific seat string

Mjlog files from Tenhou are gzipped XML. Unzip first, then pass the XML to `./out`.

## Architecture

The codebase is three Objective-C++ source files:

- **`mt19937ar.mm`** / **`mt19937ar.h`**: Mersenne Twister PRNG (`_MTRND` class). Generates the raw random numbers used to shuffle tiles.

- **`mjlog.mm`** / **`mjlog.h`**: XML log parser and data model.
  - `MjLog`: read-only model holding parsed game data (seed string, dice rolls per round, drawn tiles per round, dead wall tiles).
  - `MjLogCtrl`: mutable subclass built by `MjLogParser` as it parses.
  - `MjLogParser` (`NSXMLParserDelegate`): walks the XML elements (`SHUFFLE`, `INIT`, `AGARI`/`RYUUKYOKU`, `DORA`, tile-draw elements `T`/`U`/`V`/`W`) to reconstruct the draw order and dead wall contents.

- **`tenhou.mm`**: verification logic and `main`.
  - `setup_seed`: decodes the base64 seed from the log and initializes the MT PRNG.
  - `checkMlogRounds`: for each round, generates 137 random values, hashes them via SHA-512 (two 512-bit blocks → one shuffle array), performs a Fisher-Yates shuffle on 136 tiles, then compares dices and draw order against what the log recorded.

## Patterns & Conventions

- **NSString → std::cout**: Use `[nsstring UTF8String]` to convert NSString to C string for C++ output.
- **Property redeclaration in subclass**: When redeclaring parent's property as `readwrite` in subclass, add explicit `@synthesize propertyName;` to `@implementation` to avoid warnings.
- **XML attribute URL encoding**: Player names and other data from XML attributes are percent-encoded; decode with `stringByRemovingPercentEncoding`.
- **Use `auto`**: Use type inference to reduce clutters.
- **Build verification**: Run `sh build.sh` after changes — expect zero warnings with current `-O3 -flto` configuration.

## TODO: Known Limitations

- [ ] 3-player (sanma) mahjong support (would use 108 tiles instead of 136).
- [ ] Dora and Rinshan tile verification.
- [ ] Log format version hardcoded to `2.3` — make configurable.
