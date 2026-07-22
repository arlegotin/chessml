# Online Screenshot Benchmark v1: Attribution and Rights Evidence

Checked on 2026-07-19. This file covers the planned
`online-screenshot-v1` corpus. No screenshot was captured while preparing this
rights bundle. Each eventual capture receipt must supply its actual UTC capture
time.

The upstream-covered screenshot files are not relicensed under this
repository's root license. They retain the terms and notices of the upstream
site and assets represented in each image. The project-specific `COPYING.md`
and asset-license snapshots below govern their covered components; the generic
license texts are supplemental copies and are not substitutes for those
project-specific notices.

## Capture and preservation statement

The corpus protocol permits only a full visible browser viewport from the
public anonymous editor or the named project-authored legal/privacy page. The
planned positions are synthetic and controlled by this project. Captures must
contain no usernames, ratings, avatars, chat, notifications, account data, or
other user content.

Rendering and PNG pixels must remain unchanged: no crop, resize, compression,
overlay, blur, color adjustment, CSS injection, or other postprocessing is
allowed. Responsive scaling must come from the site's own layout. Retained PNG
bytes must be the exact bytes emitted by the browser. Nothing in this benchmark
implies sponsorship, endorsement, or official status from Lichess, PyChess,
their contributors, or the named asset authors.

## Lichess

- Live editor: <https://lichess.org/editor>
- Terms: <https://lichess.org/terms-of-service>
- Privacy: <https://lichess.org/privacy>
- Source: <https://github.com/lichess-org/lila>
- Source evidence revision:
  [`1b02f3c8754e5b389f42e6a8d9b6cedcc870ff69`](https://github.com/lichess-org/lila/commit/1b02f3c8754e5b389f42e6a8d9b6cedcc870ff69),
  committed 2026-07-19T11:16:36Z. This is a source snapshot, not a claim that
  the live deployment maps to that Git commit.
- Project notice: [`benchmarks/licenses/lichess-lila-COPYING.md`](licenses/lichess-lila-COPYING.md),
  copied byte-for-byte from the commit-pinned
  [`COPYING.md`](https://raw.githubusercontent.com/lichess-org/lila/1b02f3c8754e5b389f42e6a8d9b6cedcc870ff69/COPYING.md).
- Board: intended locked identifier `brown`. The lila notice attributes
  `public/images/board` to the lila authors and pirouetti under
  AGPL-3.0-or-later.
- Piece TASL:
  - Title: **Chessnut piece layout**
  - Author: **Alexis Luengas**
  - Source:
    <https://github.com/LexLuengas/chessnut-pieces/tree/2b8eaf14a31edad7e9deb53b1473e1d4857868a9>
  - License: **Apache License 2.0**
  - Intended locked identifier: `chessnut`
  - Retained license:
    [`benchmarks/licenses/lichess-chessnut-LICENSE.txt`](licenses/lichess-chessnut-LICENSE.txt),
    copied byte-for-byte from the immutable upstream
    [`LICENSE.txt`](https://raw.githubusercontent.com/LexLuengas/chessnut-pieces/2b8eaf14a31edad7e9deb53b1473e1d4857868a9/LICENSE.txt).
- Logo restriction: lila classifies its logo and favicon as non-free and says
  **“Only use to refer to lichess.org.”** Any logo incidentally visible in a
  full viewport is used only to identify and refer to lichess.org; it is not
  used as this benchmark's mark.
- No endorsement is asserted by Lichess, the lila authors, pirouetti, or Alexis
  Luengas.

The official Markdown alternate representations were retrieved directly from
Lichess at 2026-07-19T22:02:08Z:

- Terms response:
  <https://lichess.org/terms-of-service?lang=en&output_format=md>, HTTP 200,
  `Content-Type: text/markdown; charset=utf-8`, `Content-Length: 41867`, body
  SHA-256 `65b9f728ff41491fdccb1c604faada948caafbfc566fdddd8140382682226fd2`.
- Privacy response: <https://lichess.org/privacy?lang=en&output_format=md>,
  HTTP 200, `Content-Type: text/markdown; charset=utf-8`,
  `Content-Length: 24793`, body SHA-256
  `540ca5a61fd95cce55bf38a448d118dd997323d9fc99cfb164c1b0dabe258732`.

The retained files preserve the response bodies exactly, including their
original line endings. The terms identify a last-modified date of 2023-04-02;
the privacy policy identifies a last-updated date of 2022-03-16. The checked
terms allow use and reproduction of the site subject to the licenses governing
its component parts and require reasonable use of services. The narrow,
manual, sequential protocol described above does not introduce a material
capture prohibition found in that text.

## PyChess

- Live editor: <https://www.pychess.org/editor/chess>
- Terms: <https://www.pychess.org/terms>
- Privacy: <https://www.pychess.org/privacy>
- Source: <https://github.com/gbtami/pychess-variants>
- Observed deployed asset version: `1.11.36`. The public editor response on
  2026-07-19T21:55:18Z referenced that immutable jsDelivr tag and loaded
  `/static/pychess-variants.js?v=4d61b5d670a873d6d6d16ea3fbb7587f8ac3a91b`.
- Source tag: [`1.11.36`](https://github.com/gbtami/pychess-variants/tree/1.11.36),
  commit
  [`4d61b5d670a873d6d6d16ea3fbb7587f8ac3a91b`](https://github.com/gbtami/pychess-variants/commit/4d61b5d670a873d6d6d16ea3fbb7587f8ac3a91b),
  committed 2026-07-19T21:25:34Z.
- Project notice:
  [`benchmarks/licenses/pychess-variants-COPYING.md`](licenses/pychess-variants-COPYING.md),
  copied byte-for-byte from the tag's immutable
  [`static/COPYING.md`](https://raw.githubusercontent.com/gbtami/pychess-variants/4d61b5d670a873d6d6d16ea3fbb7587f8ac3a91b/static/COPYING.md).
- Board: intended locked identifier `brown`, source asset
  `static/images/board/8x8brown.svg`. The PyChess notice covers
  `static/images/board/*` under AGPL-3.0-or-later.
- Piece TASL:
  - Title: **Firi piece set**
  - Author: **James Faure**
  - Source: <https://github.com/jfaure/Firi-pieceset>
  - License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
  - Intended locked identifier: `firi`
  - Retained license:
    [`benchmarks/licenses/pychess-firi-LICENSE.txt`](licenses/pychess-firi-LICENSE.txt),
    copied byte-for-byte from PyChess tag `1.11.36` at
    [`static/images/pieces/firi/LICENSE.txt`](https://raw.githubusercontent.com/gbtami/pychess-variants/4d61b5d670a873d6d6d16ea3fbb7587f8ac3a91b/static/images/pieces/firi/LICENSE.txt).
- Board client: **Chessgroundx 10.7.5**, source
  <https://github.com/gbtami/chessgroundx/tree/112c6b616f9e9e6fb79528c430f602287caa3535>,
  licensed GPL-3.0 or, as its README permits, any later version. PyChess tag
  `1.11.36` resolves `chessgroundx` 10.7.5 in `yarn.lock`.
- No endorsement is asserted by PyChess, its contributors, gbtami, James
  Faure, or the Chessgroundx contributors.

The live terms and privacy pages displayed “Last updated: May 22, 2026” and
served the same deployed asset tag and revision recorded above. Their exact
official source Markdown is retained from tag `1.11.36`:

- [`benchmarks/licenses/pychess-terms-2026-05-22.md`](licenses/pychess-terms-2026-05-22.md)
  from immutable
  [`static/docs/terms.md`](https://raw.githubusercontent.com/gbtami/pychess-variants/4d61b5d670a873d6d6d16ea3fbb7587f8ac3a91b/static/docs/terms.md).
- [`benchmarks/licenses/pychess-privacy-2026-05-22.md`](licenses/pychess-privacy-2026-05-22.md)
  from immutable
  [`static/docs/privacy.md`](https://raw.githubusercontent.com/gbtami/pychess-variants/4d61b5d670a873d6d6d16ea3fbb7587f8ac3a91b/static/docs/privacy.md).

The checked PyChess terms prohibit scraping, attacks, overload, harmful
automation, and evasion of technical restrictions. The benchmark protocol is
not a scraper: it permits only separately triggered, anonymous, sequential
full-viewport captures of the exact allowlisted project pages, with no retry
loop and immediate stop on a block, rate limit, or policy change. No material
prohibition of that narrow capture was found in the checked text.

## Live asset-identity gate (not yet satisfied)

The intended identifiers above are proven by the approved design and immutable
upstream source, but they are not a substitute for observing the selected
styles in each live anonymous editor. The permitted in-app browser binding was
unavailable during this check. Therefore the following live values remain
unrecorded: each site's selected piece and board asset IDs, resolved piece
asset URL prefix, computed board-style signature, and accepted upstream
revision as one coherent five-field observation.

`benchmarks/board_recognition_online_screenshot_v1_RIGHTS.json` is
intentionally absent. No placeholder or source-only inference is allowed to
pass that machine-readable gate. Before any capture, a fresh anonymous browser
must select Chessnut/brown and Firi/brown through the normal settings UI, read
only those five identity fields per site, populate canonical `RIGHTS.json`,
and verify its evidence records against the hashes below.

## Retained evidence inventory

Every file in this table is plain text and was retained byte-for-byte. SHA-256
is over the repository file bytes.

| Path | SHA-256 | Official source |
| --- | --- | --- |
| `benchmarks/licenses/AGPL-3.0-or-later.txt` | `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0` | <https://www.gnu.org/licenses/agpl-3.0.txt> |
| `benchmarks/licenses/GPL-3.0-or-later.txt` | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` | <https://www.gnu.org/licenses/gpl-3.0.txt> |
| `benchmarks/licenses/Apache-2.0.txt` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | <https://www.apache.org/licenses/LICENSE-2.0.txt> |
| `benchmarks/licenses/CC-BY-4.0.txt` | `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411` | <https://creativecommons.org/licenses/by/4.0/legalcode.txt> |
| `benchmarks/licenses/lichess-lila-COPYING.md` | `8203edbe5691207a3ee1298e8a05a6e3371abe8d7c1ba7b788ee2c42085ca6cb` | lila commit `1b02f3c8754e5b389f42e6a8d9b6cedcc870ff69`, `COPYING.md` |
| `benchmarks/licenses/lichess-chessnut-LICENSE.txt` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | Chessnut commit `2b8eaf14a31edad7e9deb53b1473e1d4857868a9`, `LICENSE.txt` |
| `benchmarks/licenses/lichess-terms-2026-07-19.txt` | `65b9f728ff41491fdccb1c604faada948caafbfc566fdddd8140382682226fd2` | Lichess live official Markdown response, 2026-07-19T22:02:08Z |
| `benchmarks/licenses/lichess-privacy-2026-07-19.txt` | `540ca5a61fd95cce55bf38a448d118dd997323d9fc99cfb164c1b0dabe258732` | Lichess live official Markdown response, 2026-07-19T22:02:08Z |
| `benchmarks/licenses/pychess-variants-COPYING.md` | `cc4c7c2a978508743e0da2be876965f879cb789320256ee51017e1b4ef2587c5` | PyChess tag `1.11.36`, `static/COPYING.md` |
| `benchmarks/licenses/pychess-firi-LICENSE.txt` | `b0795076b6787cb92063ffd73df82a6930705dca321b249e10a9ad5175428fff` | PyChess tag `1.11.36`, `static/images/pieces/firi/LICENSE.txt` |
| `benchmarks/licenses/pychess-terms-2026-05-22.md` | `ef6af997f4c9a39626a360a0e4daa8b7ba2e8bd22ee107aad97cc1d394962931` | PyChess tag `1.11.36`, `static/docs/terms.md` |
| `benchmarks/licenses/pychess-privacy-2026-05-22.md` | `d378b8b26bcc02c3d31c2d26cd2280579df64b0eb39082f230bf39fd2a4861ff` | PyChess tag `1.11.36`, `static/docs/privacy.md` |
| `benchmarks/licenses/chessgroundx-GPL-3.0.txt` | `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903` | Chessgroundx commit `112c6b616f9e9e6fb79528c430f602287caa3535`, `LICENSE` |

The generic GNU files contain the version 3 license texts; the “or later”
choice comes from the applicable lila, PyChess, and Chessgroundx project
notices. The Chessnut upstream license happens to be byte-identical to the
retained Apache canonical text, but both paths are kept because one is the
project-specific asset record. The Firi and Chessgroundx paths likewise retain
their exact project copies even where generic license text is also present.
