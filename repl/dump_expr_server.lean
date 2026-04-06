import Mathlib
import Lean

open Lean Lean.Parser Lean.Elab Lean.Meta

partial def exprToJson (e : Expr) : String :=
  match e with
  | .bvar idx => s!"\{\"expr\": \"bvar\", \"idx\": {idx}}"
  | .fvar fv => s!"\{\"expr\": \"fvar\", \"id\": \"{fv.name}\"}"
  | .mvar mv => s!"\{\"expr\": \"mvar\", \"id\": \"{mv.name}\"}"
  | .sort lvl => s!"\{\"expr\": \"sort\", \"level\": \"{lvl}\"}"
  | .const n _ => s!"\{\"expr\": \"const\", \"name\": \"{n}\"}"
  | .app fn arg => s!"\{\"expr\": \"app\", \"fn\": {exprToJson fn}, \"arg\": {exprToJson arg}}"
  | .lam n t b _ => s!"\{\"expr\": \"lam\", \"var_name\": \"{n}\", \"var_type\": {exprToJson t}, \"body\": {exprToJson b}}"
  | .forallE n t b _ => s!"\{\"expr\": \"forallE\", \"var_name\": \"{n}\", \"var_type\": {exprToJson t}, \"body\": {exprToJson b}}"
  | .letE n t v b _ => s!"\{\"expr\": \"letE\", \"var_name\": \"{n}\", \"var_type\": {exprToJson t}, \"val\": {exprToJson v}, \"body\": {exprToJson b}}"
  | .lit (.natVal v) => s!"\{\"expr\": \"lit\", \"type\": \"nat\", \"val\": {v}}"
  | .lit (.strVal v) => s!"\{\"expr\": \"lit\", \"type\": \"str\", \"val\": \"{v}\"}"
  | .mdata _ inner => s!"\{\"expr\": \"mdata\", \"inner\": {exprToJson inner}}"
  | .proj s i inner => s!"\{\"expr\": \"proj\", \"struct\": \"{s}\", \"idx\": {i}, \"inner\": {exprToJson inner}}"

def dumpExprTree (name : Name) (typeExpr : Expr) (valExpr? : Option Expr) : IO Unit := do
  let typeJson := exprToJson typeExpr
  let valJson := match valExpr? with
    | some v => exprToJson v
    | none => "null"
  IO.println s!"\{\"theorem\": \"{name}\", \"expr_tree\": {typeJson}, \"expr_value_tree\": {valJson}}"

partial def elabLoop (inputCtx : InputContext) (pmctx : ParserModuleContext)
                     (p : ModuleParserState) (c : Command.State) : IO Command.State := do
  let (stx, p', messages) := parseCommand inputCtx pmctx p c.messages
  if p'.pos == p.pos then
    return { c with messages := messages }
  else
    let c' ← try
      let cmdCtx : Command.Context := {
        fileName := inputCtx.fileName,
        fileMap := inputCtx.fileMap,
        snap? := none,
        cancelTk? := none
      }
      let ((), newC) ← (Command.elabCommand stx).run cmdCtx
        |>.run { c with messages := messages }
        |>.toIO (fun _ => IO.userError "Command Elaboration Failed")
      pure newC
    catch e =>
      IO.eprintln s!"[Command Panic] {e}"
      pure { c with messages := messages }

    elabLoop inputCtx pmctx p' c'

def processFileExpr (fileName : String) (cachedHeader : String) (cachedBaseEnv : Environment) : IO (String × Environment) := do
  let content ← IO.FS.readFile fileName
  let inputCtx := mkInputContext content fileName
  let (header, parserState, messages) ← parseHeader inputCtx
  let headerStr := toString header.raw

  let mut baseEnv := cachedBaseEnv
  let mut currentHeader := cachedHeader

  -- Nếu header khác, tạo baseEnv MỚI SẠCH
  if headerStr != cachedHeader && headerStr != "" then
    IO.eprintln s!"[Server] Loading imports for {fileName}..."
    let (env, _) ← try
      Lean.Elab.processHeader header {} messages inputCtx
    catch e =>
      IO.eprintln s!"[HEADER FATAL] {e}"
      pure (← mkEmptyEnvironment, messages)
    baseEnv := env
    currentHeader := headerStr

  -- Khởi tạo cmdState từ baseEnv SẠCH
  let cmdState := Command.mkState baseEnv messages {}
  let pmctx : ParserModuleContext := { env := baseEnv, options := {} }

  -- Chạy elabLoop (nó sẽ đắp thêm data của file hiện tại vào)
  let finalCmdState ← elabLoop inputCtx pmctx parserState cmdState
  let envWithFileDecls := finalCmdState.env

  for msg in finalCmdState.messages.toList do
    let msgStr ← msg.toString
    IO.eprintln s!"[Compiler Msg] {msgStr}"

  -- Lấy các constant MỚI thêm vào trong module này (nhờ map₂)
  let localDecls := envWithFileDecls.constants.map₂

  localDecls.forM fun name cinfo => do
    let val? := match cinfo with
      | .thmInfo val => some val.value
      | .defnInfo val => some val.value
      | _ => none
    dumpExprTree name cinfo.type val?

  IO.println "===EOF==="
  (← IO.getStdout).flush

  -- QUAN TRỌNG NHẤT: Trả về baseEnv (chỉ chứa Mathlib), KHÔNG trả về envWithFileDecls
  return (currentHeader, baseEnv)

partial def serverLoop (stdin : IO.FS.Stream) (cachedHeader : String) (cachedBaseEnv : Environment) : IO Unit := do
  let line ← stdin.getLine
  if line == "" then return ()

  let fileName := line.trimAscii.toString
  let mut nextHeader := cachedHeader
  let mut nextBaseEnv := cachedBaseEnv

  if fileName != "" then
    try
      -- Truyền và nhận lại baseEnv SẠCH
      let (h, baseE) ← processFileExpr fileName cachedHeader cachedBaseEnv
      nextHeader := h
      nextBaseEnv := baseE
    catch e =>
      IO.eprintln s!"[Expr Server Panic] {e}"
      IO.println "===EOF==="
      (← IO.getStdout).flush

  serverLoop stdin nextHeader nextBaseEnv

def main : IO Unit := do
  initSearchPath (← Lean.findSysroot)
  let emptyEnv ← mkEmptyEnvironment
  let stdin ← IO.getStdin

  IO.eprintln "[Server] Lean Expr Tree Dump Server is ready!"
  serverLoop stdin "" emptyEnv
