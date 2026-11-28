"""Pure-Python package runner controls for ordinal and forwarded-ordinal imports."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pe_run_package import plan  # noqa: E402


class PackageOrdinalPlanChecks(unittest.TestCase):
    def report(self, status="bundled_export_present"):
        return dict(entry="app.exe", reachable={"app.exe": {}, "shim.dll": {}, "core.dll": {}},
                    imports=[
                        dict(image="app.exe", dll="shim.dll", symbol="Start", kind="normal", status=status,
                             chain=[dict(provider="shim.dll", symbol="Start", dll="shim.dll", package=True),
                                    dict(provider="core.dll", symbol="#77", dll="core.dll", package=True)]),
                        dict(image="app.exe", dll="core.dll", symbol="#77", kind="delay", status=status,
                             chain=[dict(provider="core.dll", symbol="#77", dll="core.dll", package=True)]),
                    ])

    def test_direct_and_forwarded_ordinals_are_bound_and_ordered(self):
        order, bindings = plan(self.report())
        self.assertEqual(order, ["core.dll", "shim.dll", "app.exe"])
        self.assertEqual(bindings["app.exe"][("core.dll", "#77")]["symbol"], "#77")
        self.assertEqual(bindings["app.exe"][("shim.dll", "Start")]["provider"], "core.dll")
        self.assertEqual(bindings["app.exe"][("shim.dll", "Start")]["symbol"], "#77")

    def test_missing_local_ordinal_fails_before_attachment(self):
        for status in ("bundled_export_not_found", "bundled_forwarder_cycle"):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, status):
                plan(self.report(status))

    def test_external_ordinal_is_not_assumed_missing(self):
        report = dict(entry="app.exe", reachable={"app.exe": {}},
                      imports=[dict(image="app.exe", kind="normal", dll="system.dll", symbol="#77",
                                    status="external_unverified",
                                    chain=[dict(provider="system.dll", dll="system.dll", symbol="#77",
                                                package=False)])])
        order, bindings = plan(report)
        self.assertEqual(order, ["app.exe"])
        self.assertEqual(bindings["app.exe"][("system.dll", "#77")]["symbol"], "#77")


if __name__ == "__main__":
    unittest.main()
