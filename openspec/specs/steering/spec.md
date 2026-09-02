# Steering Specification

## Purpose

Lets a trace be generated with a chosen SAE feature's direction added to a
layer's residual stream during generation, so the effect of amplifying or
suppressing a specific, human-labeled concept can be observed in the
resulting trace.

## Requirements

### Requirement: Steering targets one labeled feature at one layer
A steering specification SHALL consist of exactly one layer, one feature
index within that layer's SAE, and one coefficient, applied throughout
generation.

#### Scenario: Single feature steering
- **WHEN** a trace is generated with a steering specification naming layer
  `L`, feature `F`, and coefficient `c`
- **THEN** the resulting trace reflects generation influenced by that one
  `(L, F, c)` intervention and no other

### Requirement: Steering intervenes on the residual stream during generation
Steering SHALL add the named feature's decoder direction, scaled by the
coefficient, to the residual stream at the named layer at every position
produced during generation, before that position's residual is captured.

#### Scenario: Captured residuals reflect the intervention
- **WHEN** a trace is generated with steering active at layer `L`
- **THEN** the residual captured at layer `L` for each generated position
  reflects the added direction, and any layer or pass that reads that
  residual (SAE features, logit lens, attribution) sees the steered value,
  not the unsteered one

#### Scenario: Zero coefficient is a no-op
- **WHEN** a trace is generated with a steering specification whose
  coefficient is zero
- **THEN** the resulting trace's residuals and completion match an
  unsteered trace generated from the same prompt and length

### Requirement: An unsteered trace is unaffected by steering's existence
Requesting an ordinary trace SHALL produce the same result as it did before
steering existed; steering SHALL only take effect when explicitly
requested.

#### Scenario: Plain trace request is unaffected
- **WHEN** a trace is generated without a steering specification
- **THEN** its residuals, completion, and every enrichment pass's output
  are identical to what they would be if steering did not exist

### Requirement: A steered trace records the steering that produced it
A trace produced under steering SHALL record the layer, feature index, and
coefficient used, so a reader of the trace can tell it was steered and
with what.

#### Scenario: Steering parameters are visible on the trace
- **WHEN** a client retrieves a trace that was generated with steering
- **THEN** the trace document includes the layer, feature index, and
  coefficient that were applied during its generation
