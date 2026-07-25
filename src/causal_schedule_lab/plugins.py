from __future__ import annotations

from typing import Any, Callable

from .project import ProjectContext, load_object
from .repair import ConditionalRepairGenerator, GenericCPSATRepairGenerator


GeneratorFactory = Callable[[ProjectContext], ConditionalRepairGenerator]
DomainOracle = Callable[[Any, Any, Any], dict[str, Any]]


def build_generator(context: ProjectContext) -> ConditionalRepairGenerator:
    """Resolve a repair engine without introducing domain logic into the core."""

    name = context.manifest.generator
    options = dict(context.manifest.generator_options)
    if name == "generic-cp-sat":
        return GenericCPSATRepairGenerator(**options)
    factory = load_object(name)
    factory = factory() if isinstance(factory, type) else factory
    return factory(context)


def build_domain_oracle(context: ProjectContext) -> DomainOracle | None:
    specification = context.manifest.domain_oracle
    if not specification:
        return None
    factory = load_object(specification)
    factory = factory() if isinstance(factory, type) else factory
    return factory(context)
