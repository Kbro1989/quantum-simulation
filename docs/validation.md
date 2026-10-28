# Validation

`python benchmarks/validation_suite.py` runs every check below against an exact, analytic or independent numerical reference and writes the report reproduced on this page. The suite runs in continuous integration on every push and pull request. A tolerance tied to floating-point roundoff (for example norm conservation) is derived from the number of operations rather than fixed, because rounding differs between CPU architectures.

In addition, the unit-test suite contains more than 100 physics and API tests, and the GPU path has been verified on an NVIDIA Tesla T4 (see [GPU acceleration](guide/gpu.md)).

---

--8<-- "benchmarks/validation_report.md:3:"
