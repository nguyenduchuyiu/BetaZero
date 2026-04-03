import Mathlib
import Lean

open Lean Lean.Parser

partial def extractBlocks (stx : Syntax) : IO Unit := do
  let kindStr := toString stx.getKind
  match stx.getPos?, stx.getTailPos? with
  | some startP, some tailP =>
    let jsonStr := s!"\{\"kind\": \"{kindStr}\", \"start_byte\": {startP.byteIdx}, \"end_byte\": {tailP.byteIdx}}"
    IO.println jsonStr
  | _, _ => pure ()
  for arg in stx.getArgs do
    extractBlocks arg

partial def parseLoop (inputCtx : InputContext) (pmctx : ParserModuleContext) (p : ModuleParserState) (m : MessageLog) : IO Unit := do
  let (stx, p', m') := parseCommand inputCtx pmctx p m
  extractBlocks stx
  if p'.pos == p.pos then return ()
  else parseLoop inputCtx pmctx p' m'

def processFile (fileName : String) (lastHeader : String) (lastEnv : Environment) : IO (String × Environment) := do
  let content ← IO.FS.readFile fileName
  let inputCtx := mkInputContext content fileName
  let (header, parserState, messages) ← parseHeader inputCtx
  let headerStr := toString header.raw

  let mut currentEnv := lastEnv
  let mut newHeader := lastHeader

  -- [ĐÃ FIX]: Chỉ nạp lại Environment khi có Import MỚI và KHÁC RỖNG
  -- Nếu file đéo có import (headerStr == ""), nó sẽ bốc luôn cái currentEnv (Mathlib) ra dùng!
  if headerStr != lastHeader && headerStr != "" then
    IO.eprintln s!"[Server] New imports detected. Loading Environment..."
    let (env, _) ← try
      Lean.Elab.processHeader header {} messages inputCtx
    catch _ =>
      pure (← mkEmptyEnvironment, messages)
    currentEnv := env
    newHeader := headerStr
  else
    -- Kế thừa toàn bộ Mathlib từ file Warmup
    pure ()

  let pmctx : ParserModuleContext := { env := currentEnv, options := {} }

  extractBlocks header.raw
  parseLoop inputCtx pmctx parserState messages

  IO.println "===EOF==="
  (← IO.getStdout).flush

  return (newHeader, currentEnv)

partial def serverLoop (stdin : IO.FS.Stream) (lastHeader : String) (lastEnv : Environment) : IO Unit := do
  let line ← stdin.getLine
  if line == "" then return ()

  let fileName := (line.replace "\n" "").replace "\r" ""
  let mut nextHeader := lastHeader
  let mut nextEnv := lastEnv

  if fileName != "" then
    try
      let (h, e) ← processFile fileName lastHeader lastEnv
      nextHeader := h
      nextEnv := e
    catch e =>
      IO.eprintln s!"[Error] Failed to process {fileName}: {e}"
      IO.println "===EOF==="
      (← IO.getStdout).flush

  serverLoop stdin nextHeader nextEnv

def main : IO Unit := do
  initSearchPath (← Lean.findSysroot)

  -- [ĐÃ FIX]: Khởi đầu với tay trắng (mkEmptyEnvironment), đéo nạp Mathlib thừa thãi ở main nữa!
  let emptyEnv ← mkEmptyEnvironment

  let stdin ← IO.getStdin
  IO.eprintln "[Server] Lean AST Server is ready!"

  -- Gieo môi trường rỗng vào. Lúc Warmup gửi file "import Mathlib" tới, server sẽ tự độ lại Environment.
  serverLoop stdin "" emptyEnv
