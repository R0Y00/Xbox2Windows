#!/usr/bin/env python3
"""Checks for tools/gdk_platform_profiles.py.

Builds a small MSBuild platform tree in a temporary directory. No GDK or
Windows is required.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gdk_platform_profiles as profiles  # noqa: E402


NAMESPACE = 'xmlns="http://schemas.microsoft.com/developer/msbuild/2003"'


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<Project {NAMESPACE}>\n{body}\n</Project>\n", encoding="utf-8"
    )


def make_tree(root):
    platform = (
        root
        / "VS2022"
        / "flatDeployment"
        / "MSBuild"
        / "Microsoft"
        / "VC"
        / "v170"
        / "Platforms"
        / "Gaming.Xbox.Scarlett.x64"
    )
    write(
        platform / "261000" / "Platform.Common.props",
        """  <PropertyGroup>
    <UseDebugLibraries Condition="'$(Configuration)'=='Debug'">true</UseDebugLibraries>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <ItemDefinitionGroup>
    <ClCompile>
      <PreprocessorDefinitions>_GAMING_XBOX;_GAMING_XBOX_SCARLETT;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <AdditionalIncludeDirectories>$(XdkEditionRootVS2022)Include;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
      <AdditionalOptions>/favor:AMD64 %(AdditionalOptions)</AdditionalOptions>
      <FloatingPointModel>Fast</FloatingPointModel>
    </ClCompile>
    <Link>
      <AdditionalDependencies>xgameplatform.lib</AdditionalDependencies>
      <TargetMachine>MachineX64</TargetMachine>
    </Link>
  </ItemDefinitionGroup>""",
    )
    write(
        platform / "Platform.props",
        """  <PropertyGroup>
    <XdkEditionRootVS2022>D:\\gdk\\</XdkEditionRootVS2022>
  </PropertyGroup>""",
    )
    write(platform / "Platform.targets", "  <PropertyGroup><XdkEdition>261000</XdkEdition></PropertyGroup>")
    write(
        platform / "PlatformToolsets" / "v143" / "Toolset.props",
        """  <ItemDefinitionGroup>
    <ClCompile>
      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>
    </ClCompile>
  </ItemDefinitionGroup>""",
    )
    return platform


def check(condition, what):
    if not condition:
        raise AssertionError(what)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        make_tree(root)

        found = profiles.choose(root, "Gaming.Xbox.Scarlett.x64", "v170", "261000")
        check(found is not None, "platform not found")
        check(found.name == "Gaming.Xbox.Scarlett.x64", f"platform {found.name}")

        listed = profiles.main(["list", "--root", str(root)])
        check(listed == 0, f"list exit {listed}")

        output = root / "profile.json"
        code = profiles.main([
            "scan", "--root", str(root), "--platform", "Gaming.Xbox.Scarlett.x64",
            "--json", str(output),
        ])
        check(code == 0, f"scan exit {code}")
        document = json.loads(output.read_text(encoding="utf-8"))
        check(document["platform"] == "Gaming.Xbox.Scarlett.x64", "platform name")
        check(document["defines"] == ["_GAMING_XBOX", "_GAMING_XBOX_SCARLETT"], f"defines {document['defines']}")
        check(document["includes"] == ["D:\\gdk\\Include"], f"includes {document['includes']}")
        check(document["dependencies"] == ["xgameplatform.lib"], f"deps {document['dependencies']}")
        check(document["flags"]["floating_point_model"] == "Fast", "floating point")
        check(document["flags"]["runtime_library"] == "MultiThreadedDLL", "runtime library")
        check(document["flags"]["target_machine"] == "MachineX64", "target machine")
        check(document["flags"]["character_set"] == "Unicode", "character set")
        conditions = document["properties"]["UseDebugLibraries"]
        check(conditions[0]["condition"] == "'$(Configuration)'=='Debug'", f"condition {conditions}")
        check(len(document["sources"]) == 4, f"sources {len(document['sources'])}")

        missing = profiles.main([
            "scan", "--root", str(root), "--platform", "Gaming.Desktop.x64",
            "--json", str(root / "missing.json"),
        ])
        check(missing == 2, f"missing platform exit {missing}")
    print("gdk_platform_profiles_check: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
