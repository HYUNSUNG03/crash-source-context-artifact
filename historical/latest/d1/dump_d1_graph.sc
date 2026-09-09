// Import a small C/C++ source set and dump the graph needed by D1.
@main def exec(): Unit = {
  import java.nio.file.{Files, Paths}
  import java.nio.charset.StandardCharsets
  import io.shiftleft.codepropertygraph.generated.nodes.*

  val srcDir = sys.env.getOrElse("D1_SRC", sys.error("D1_SRC is not set"))
  val outFile = sys.env.getOrElse("D1_OUT", sys.error("D1_OUT is not set"))
  val project = sys.env.getOrElse("D1_PROJECT", "d1-fixtures")
  val fileSuffixes = sys.env.get("D1_FILE_SUFFIXES").toSeq
    .flatMap(_.split(";")).map(_.trim.replace("\\", "/")).filter(_.nonEmpty)
  sys.env.get("D1_CPG") match {
    case Some(cpgFile) => importCpg(cpgFile, projectName = project)
    case None          => importCode(inputPath = srcDir, projectName = project)
  }

  def esc(s: String): String =
    Option(s).getOrElse("").replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "")

  def nameOf(n: StoredNode): String = n match {
    case i: Identifier        => i.name
    case l: Local             => l.name
    case p: MethodParameterIn => p.name
    case c: Call              => c.name
    case m: Method            => m.name
    case _                    => ""
  }

  def codeOf(n: StoredNode): String = n match {
    case a: AstNode => a.code
    case _          => ""
  }

  def info(n: StoredNode): String = {
    val line = n match { case a: AstNode => a.lineNumber.getOrElse(-1); case _ => -1 }
    val column = n match { case a: AstNode => a.columnNumber.getOrElse(-1); case _ => -1 }
    val order = n match { case a: AstNode => a.order; case _ => -1 }
    val argumentIndex = n match { case e: Expression => e.argumentIndex; case _ => -1 }
    s"""{"id":${n.id},"label":"${n.label}","line":$line,"column":$column,"order":$order,"argument_index":$argumentIndex,"name":"${esc(nameOf(n))}","code":"${esc(codeOf(n))}"}"""
  }

  val methods = cpg.method.l.filter { m =>
    val filename = m.filename.replace("\\", "/")
    !m.isExternal && (fileSuffixes.isEmpty || fileSuffixes.exists(filename.endsWith))
  }
  val nodes = methods.flatMap(_.ast.l).groupBy(_.id).values.map(_.head).toSeq
  val nodeIds = nodes.map(_.id).toSet

  def edges(label: String): String = nodes.flatMap { src =>
    src.out(label).l.collect {
      case dst: StoredNode if nodeIds.contains(dst.id) =>
        s"""{"src":${src.id},"dst":${dst.id}}"""
    }
  }.mkString(",")

  val methodJson = methods.map { method =>
    val ids = method.ast.l.map(_.id).mkString(",")
    s"""{"id":${method.id},"name":"${esc(method.name)}","file":"${esc(method.filename.replace("\\", "/"))}","node_ids":[$ids]}"""
  }.mkString(",")

  val json = s"""{"methods":[$methodJson],"nodes":[${nodes.map(info).mkString(",")}],"edges":{"AST":[${edges("AST")}],"REF":[${edges("REF")}],"CFG":[${edges("CFG")}],"CDG":[${edges("CDG")}],"REACHING_DEF":[${edges("REACHING_DEF")}],"CALL":[${edges("CALL")}]}}"""
  Files.write(Paths.get(outFile), json.getBytes(StandardCharsets.UTF_8))
  println("wrote " + outFile + " methods=" + methods.size + " nodes=" + nodes.size)
}
