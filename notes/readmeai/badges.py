"""
Methods to generate and format svg badges for the README file.
"""

from typing import Tuple

from notes.config.config_loader import ConfigLoader
from notes.options import BadgeOptions
from notes.services.git import GitHost
from notes.utils.file_handler import FileHandler
from notes.utils.file_resources import get_resource_path

_package = "readmeai.generators"
_submodule = "assets"


def _format_badges(badges: list[str]) -> str:
    """Format SVG badge icons as HTML."""
    total = len(badges)
    if badges is None or total == 0:
        return ""

    badges_per_line = total if total < 9 else (total // 2) + (total % 2)

    lines = []
    for i in range(0, total, badges_per_line):
        line = "\n\t".join(
            [
                f'<img src="{badge}" alt="{badge.split("/badge/")[1].split("-")[0]}">'
                for badge in badges[i : i + badges_per_line]
            ]
        )
        lines.append(
            f"{line}\n\t<br>" if i + badges_per_line < total else f"{line}\n"
        )

    return "\n\t".join(lines)


def build_default_badges(
    config: ConfigLoader, full_name: str, host: str
) -> str:
    """Build metadata badges using shields.io."""
    return config.md.shields_icons.format(
        host=host,
        full_name=full_name,
        badge_color=config.md.badge_color,
        badge_style=config.md.badge_style,
    )


def build_project_badges(
    dependencies: list[str], icons: dict[str, str], style: str
) -> str:
    """Build HTML badges for project dependencies."""
    badges = [
        icons[str(dependency).lower()]
        for dependency in dependencies
        if str(dependency).lower() in icons
    ]

    # Sort badges by hex value (from light to dark color)
    badges.sort(key=lambda b: int(b[1], 16) if b[1] else 0, reverse=True)
    badges = [badge[0].format(style) for badge in badges]
    return _format_badges(badges)


def shields_icons(
    conf: ConfigLoader, dependencies: list, full_name: str, git_host: str
) -> Tuple[str, str]:
    """
    Generates badges for the README using shields.io icons.
    """
    icons_path = get_resource_path(
        conf.files.shields_icons,
        _package,
        _submodule,
    )
    icons_dict = FileHandler().read(icons_path)

    default_icons = build_default_badges(conf, full_name, git_host)

    project_badges = build_project_badges(
        dependencies, icons_dict, conf.md.badge_style
    )
    project_badges = conf.md.badge_icons.format(
        alignment=conf.md.alignment, badge_icons=project_badges
    )

    if (
        conf.md.badge_style == BadgeOptions.DEFAULT.value
        and git_host != GitHost.LOCAL
    ):
        return (
            default_icons,
            "<!-- default option, no dependency badges. -->\n",
        )

    if git_host == GitHost.LOCAL:
        return (
            "<!-- local repository, no metadata badges. -->\n",
            project_badges,
        )

    return default_icons, project_badges


def skill_icons(conf: ConfigLoader, dependencies: list) -> str:
    """
    Generates badges for the README using skill icons, from the
    repository - https://github.com/tandpfun/skill-icons.
    """
    dependencies.extend(["md"])

    icons_path = get_resource_path(
        conf.files.skill_icons, _package, _submodule
    )
    icons_dict = FileHandler().read(icons_path)

    skill_icons = [
        icon for icon in icons_dict["icons"]["names"] if icon in dependencies
    ]
    skill_icons = ",".join(skill_icons)
    skill_icons = icons_dict["url"]["base_url"] + skill_icons

    if conf.md.badge_style == "skills-light":
        skill_icons = f"{skill_icons}&theme=light"

    conf.md.skill_icons = conf.md.skill_icons.format(skill_icons)

    return conf.md.badge_icons.format(
        alignment=conf.md.alignment, badge_icons=conf.md.skill_icons
    )
