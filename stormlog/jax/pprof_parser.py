import gzip
from typing import Any, Dict, List

try:
    from . import profile_pb2
except ImportError:
    raise ImportError(
        "Could not import profile_pb2. Please run: \n"
        "curl -sO https://raw.githubusercontent.com/google/pprof/master/proto/profile.proto && "
        "python -m grpc_tools.protoc -I. --python_out=. profile.proto"
    )


def parse_jax_memory_profile(file_path: str) -> Dict[str, Any]:
    """Parse a JAX .prof (gzipped pprof protobuf) using the official protobuf schema."""
    with gzip.open(file_path, "rb") as f:
        data = f.read()

    profile = profile_pb2.Profile()  # type: ignore
    profile.ParseFromString(data)

    string_table = profile.string_table

    # Map function_id -> function name
    functions: Dict[int, str] = {}
    for func in profile.function:
        functions[func.id] = string_table[func.name]

    # Map location_id -> [function_names]
    locations: Dict[int, List[str]] = {}
    for loc in profile.location:
        names = []
        for line in loc.line:
            func_name = functions.get(line.function_id, "<unknown>")
            names.append(func_name)
        locations[loc.id] = names

    # Flatten samples
    samples = []
    for sample in profile.sample:
        # pprof puts innermost call first, so reverse to get root->leaf stack
        stack = []
        for loc_id in sample.location_id:
            loc_names = locations.get(loc_id, ["<unknown>"])
            # The line entries in a location are innermost-first too
            stack.extend(loc_names)

        stack.reverse()
        samples.append({"stack": stack, "values": list(sample.value)})

    return {"samples": samples}
