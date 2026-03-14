-- repl/dump_ast.lean

import Mathlib

open Lean
open Lean.Parser


/--
  In ra thông tin của node AST: kind, start_byte, end_byte (dạng JSON).
  Đệ quy qua toàn bộ cây.
-/
partial def extractBlocks (stx : Syntax) : IO Unit := do
  let kindStr := toString stx.getKind
  match stx.getPos?, stx.getTailPos? with
  | some startP, some tailP =>
    let jsonStr :=
      "{" ++
      "\"kind\": \"" ++ kindStr ++ "\", " ++
      "\"start_byte\": " ++ toString startP.byteIdx ++ ", " ++
      "\"end_byte\": " ++ toString tailP.byteIdx ++
      "}"
    IO.println jsonStr
  | _, _ => pure ()
  -- Đệ quy qua các argument con
  for arg in stx.getArgs do
    extractBlocks arg

/--
  parseLoop: Đọc tới hết các câu lệnh trong file và in AST.
-/
partial def parseLoop
    (inputCtx : InputContext)
    (pmctx : ParserModuleContext)
    (p : ModuleParserState)
    (m : MessageLog)
    : IO Unit := do
  let (stx, p', m') := parseCommand inputCtx pmctx p m
  extractBlocks stx
  if p'.pos == p.pos then
    return ()
  else
    parseLoop inputCtx pmctx p' m'

/--
  main: Nhận file Lean, in AST của header và từng câu lệnh (toàn bộ).
-/
def main (args : List String) : IO Unit := do
  initSearchPath (← Lean.findSysroot)

  let t0 ← IO.monoMsNow

  let fileName := args[0]!
  let content ← IO.FS.readFile fileName
  let inputCtx := mkInputContext content fileName

  let t1 ← IO.monoMsNow
  IO.println s!"Read file: {t1 - t0} ms"

  let (header, parserState, messages) ← parseHeader inputCtx

  let t2 ← IO.monoMsNow
  IO.println s!"Parse header: {t2 - t1} ms"

  let (env, messages) ←
    try
      Lean.Elab.processHeader header {} messages inputCtx
    catch _ =>
      let env ← mkEmptyEnvironment
      pure (env, messages)

  let t3 ← IO.monoMsNow
  IO.println s!"Process header (LOAD MATHLIB): {t3 - t2} ms"

  let pmctx : ParserModuleContext := { env := env, options := {} }

  parseLoop inputCtx pmctx parserState messages

  let t4 ← IO.monoMsNow
  IO.println s!"Parse commands: {t4 - t3} ms"
