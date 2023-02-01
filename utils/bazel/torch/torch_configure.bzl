"""local torch configure bzl file"""

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

def _is_windows(repository_ctx):
    """Returns true if the host operating system is windows."""
    os_name = repository_ctx.os.name.lower()
    if os_name.find("windows") != -1:
        return True
    return False

def _fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sPython Configuration Error:%s %s\n" % (red, no_color, msg))

def _get_python_bin(repository_ctx):
    """Gets the python bin path."""
    python_bin = repository_ctx.os.environ.get(_PYTHON_BIN_PATH)
    if python_bin != None:
        return python_bin

    python_short_name = "python" + repository_ctx.attr.python_version
    if _is_windows(repository_ctx):
        python_short_name = "python"
    python_bin_path = repository_ctx.which(python_short_name)

    if python_bin_path != None:
        return str(python_bin_path)
    _fail("Cannot find python in PATH, please make sure " +
          "python is installed and add its directory in PATH, or --define " +
          "%s='/something/else'.\nPATH=%s" % (
              _PYTHON_BIN_PATH,
              repository_ctx.os.environ.get("PATH", ""),
          ))

def _create_local_torch_repository(repository_ctx):
    python_bin = _get_python_bin(repository_ctx)
    print(python_bin)
    torch_path = repository_ctx.execute([
        python_bin,
        "-c",
        "import torch.utils;" +
        "print(torch.__path__[0])",
    ])
    torch_path = torch_path.stdout.splitlines()[0]
    if _is_windows(repository_ctx):
        torch_path = torch_path.replace("\\", "/")

    if torch_path:
        print(torch_path)
        repository_ctx.symlink(torch_path, "torch")
        repository_ctx.template(
            "BUILD",
            Label("//torch:torch.tpl"),
        )
    else:
        print("torch_path not exists", torch_path)
    return torch_path

torch_configure = repository_rule(
    implementation = _create_local_torch_repository,
    environ = [
        _PYTHON_BIN_PATH,
    ],
    attrs = {
        "python_version": attr.string(default = "3"),
    },
)
