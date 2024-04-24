"""Abstract factory module for all project file core."""

from typing import Dict, Type

from notes.parsers.parser import BaseFileParser
from notes.parsers.configuration.docker import DockerComposeParser, DockerfileParser
from notes.parsers.configuration.properties import PropertiesParser
from notes.parsers.language.cpp import CMakeParser, MakefileAmParser, ConfigureAcParser
from notes.parsers.language.go import GoModParser
from notes.parsers.language.python import TomlParser, RequirementsParser, YamlParser
from notes.parsers.language.rust import CargoTomlParser
from notes.parsers.language.swift import SwiftPackageParser
from notes.parsers.package.gradle import BuildGradleKtsParser, BuildGradleParser
from notes.parsers.package.maven import MavenParser
from notes.parsers.package.npm import PackageJsonParser, YarnLockParser

ParserRegistryType = dict[str, Type[BaseFileParser]]

PARSER_REGISTRY = {
    # Configuration
    ".properties": PropertiesParser,
    # Language/Framework
    # Python
    "Pipfile": TomlParser(),
    "pyproject.toml": TomlParser(),
    "requirements.in": RequirementsParser(),
    "requirements.txt": RequirementsParser(),
    "requirements-dev.txt": RequirementsParser(),
    "requirements-test.txt": RequirementsParser(),
    "requirements-prod.txt": RequirementsParser(),
    "dev-requirements.txt": RequirementsParser(),
    "environment.yml": YamlParser(),
    "environment.yaml": YamlParser(),
    # "setup.py": setup_py_parser,
    # "setup.cfg": setup_cfg_parser,
    # C/C++
    "cmakeLists.txt": CMakeParser(),
    "configure.ac": ConfigureAcParser(),
    "Makefile.am": MakefileAmParser(),
    # JavaScript/Node.js
    "package.json": PackageJsonParser(),
    "yarn.lock": YarnLockParser(),
    # Kotlin and Kotlin DSL
    "build.gradle": BuildGradleParser(),
    "build.gradle.kts": BuildGradleKtsParser(),
    # Go
    "go.mod": GoModParser(),
    # Java
    "pom.xml": MavenParser(),
    # Rust
    "cargo.toml": CargoTomlParser(),
    # Swift
    "Package.swift": SwiftPackageParser(),
    "Dockerfile": DockerfileParser(),
    "docker-compose.yaml": DockerComposeParser(),
    # Package Managers
    # Monitoring and Logging
}


def parser_handler() -> Dict[str, BaseFileParser]:
    """Returns a dictionary of callable file parsers methods."""
    return PARSER_REGISTRY
