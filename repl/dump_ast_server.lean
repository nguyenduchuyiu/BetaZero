import Mathlib
import Lean

open Lean Lean.Parser

/-- Trích xuất Node AST và in ra JSON phẳng -/
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

/-- Xử lý file với cơ chế Smart Cache Environment -/
def processFile (fileName : String) (lastHeader : String) (lastEnv : Environment) : IO (String × Environment) := do
  let content ← IO.FS.readFile fileName
  let inputCtx := mkInputContext content fileName
  let (header, parserState, messages) ← parseHeader inputCtx

  -- [GIẢI PHÁP TỐI THƯỢNG]: Chuyển AST của Header thành chuỗi làm Cache Key
  -- Né hoàn toàn Dependent Type của String Slice trong bản Lean mới
  let headerStr := toString header.raw

  let mut currentEnv := lastEnv
  let mut newHeader := lastHeader

  -- Nếu Header khác với file trước đó -> Bắt buộc nạp lại Environment (Mất 20s+)
  if headerStr != lastHeader || lastHeader == "" then
    IO.eprintln s!"[Server] New imports detected. Loading Environment..."
    let (env, _) ← try
      Lean.Elab.processHeader header {} messages inputCtx
    catch _ =>
      pure (← mkEmptyEnvironment, messages)
    currentEnv := env
    newHeader := headerStr
  else
    -- Nếu Header y xì đúc -> FAST PATH (0ms) DÙNG LẠI MÔI TRƯỜNG CŨ!
    pure ()

  -- Nhờ currentEnv này, Parser sẽ hiểu các Macro của Mathlib (linarith, ring...)
  let pmctx : ParserModuleContext := { env := currentEnv, options := {} }

  extractBlocks header.raw
  parseLoop inputCtx pmctx parserState messages

  IO.println "===EOF==="
  (← IO.getStdout).flush

  return (newHeader, currentEnv)

/-- Vòng lặp Server truyền Trạng thái Cache -/
partial def serverLoop (stdin : IO.FS.Stream) (lastHeader : String) (lastEnv : Environment) : IO Unit := do
  let line ← stdin.getLine
  if line == "" then return ()

  let fileName := (line.replace "\n" "").replace "\r" ""
  let mut nextHeader := lastHeader
  let mut nextEnv := lastEnv

  if fileName != "" then
    try
      let t0 ← IO.monoMsNow
      -- Truyền Cache qua từng vòng lặp
      let (h, e) ← processFile fileName lastHeader lastEnv
      nextHeader := h
      nextEnv := e
      let t1 ← IO.monoMsNow
      IO.eprintln s!"[Server] Processed {fileName} in {t1 - t0} ms"
    catch e =>
      IO.eprintln s!"[Error] Failed to process {fileName}: {e}"
      IO.println "===EOF==="
      (← IO.getStdout).flush

  serverLoop stdin nextHeader nextEnv

def main : IO Unit := do
  initSearchPath (← Lean.findSysroot)

  let imports : Array Import := #[{ module := `Mathlib : Import }]
  let baseEnv ← importModules imports {}

  let stdin ← IO.getStdin
  IO.eprintln "[Server] Lean AST Server is ready!"

  -- Khởi động Server, gieo baseEnv vào làm vốn mồi
  serverLoop stdin "" baseEnv
