import subprocess
import pathlib
from androguard.core import apk
from lxml import etree
import xml.etree.ElementTree as ET


class CodeCoverageUtility:
    def __init__(self, apk_path, adb_path, device_name, jacoco_cli_path, logger):
        self.apk_path = apk_path
        self.adb_path = adb_path
        self.device_name = device_name
        self.jacoco_cli_path = jacoco_cli_path
        self.logger = logger

    def check_instrumentation(self):
        a = apk.APK(self.apk_path)
        instr = a.get_instrumentation()
        if instr:
            self.logger.info(f"Instrumentation found: {instr}")
        else:
            self.logger.warning("Instrumentation info not found in APK.")
        return instr is not None, a.package

    def generate_coverage_ec(self, package_name):
        try:
            result = subprocess.run([
                self.adb_path, "-s", self.device_name,
                "shell", "am", "instrument", "-w", "-e", "coverage", "true",
                f"{package_name}.test/androidx.test.runner.AndroidJUnitRunner"
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.logger.info("Instrumentation run completed.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Instrumentation failed: {e.stderr.decode().strip()}")
            return None

        coverage_file = f"app/result/{package_name}_coverage.ec"
        try:
            subprocess.run([
                self.adb_path, "-s", self.device_name,
                "pull", f"/data/user/0/{package_name}/coverage.ec", coverage_file
            ], check=True)
            self.logger.info(f"Coverage file pulled to: {coverage_file}")
            return coverage_file
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to pull coverage file: {e.stderr.decode().strip()}")
            return None

    def convert_coverage_to_xml(self, coverage_file):
        try:
            coverage_dir = pathlib.Path(coverage_file).parent
            report_dir = coverage_dir / "coverage_report"
            report_dir.mkdir(parents=True, exist_ok=True)
            xml_path = report_dir / "coverage.xml"

            subprocess.run([
                "java", "-jar", self.jacoco_cli_path, "report", coverage_file,
                "--classfiles", f"{self.apk_path}/build/intermediates/javac/debug/classes/",
                "--sourcefiles", f"{self.apk_path}/src/main/java",
                "--html", str(report_dir),
                "--xml", str(xml_path)
            ], check=True)

            self.logger.info(f"XML coverage report generated at: {xml_path}")
            return str(xml_path)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to generate XML report: {e.stderr.decode().strip()}")
            return None

    def parse_coverage_percentage(self, xml_path):
        try:
            tree = etree.parse(xml_path)
            counter = tree.xpath("//counter[@type='INSTRUCTION']")
            if counter:
                covered = int(counter[0].get("covered"))
                missed = int(counter[0].get("missed"))
                total = covered + missed
                percent = (covered / total) * 100 if total > 0 else 0.0
                return round(percent, 2)
        except Exception as e:
            self.logger.error(f"Error parsing XML: {str(e)}")
        return 0.0

    def run(self):
        has_instrumentation, package_name = self.check_instrumentation()
        if not has_instrumentation:
            self.logger.error("APK does not contain instrumentation. Please rebuild with testInstrumentationRunner.")
            return 0.0

        coverage_file = self.generate_coverage_ec(package_name)
        if not coverage_file:
            return 0.0

        xml_path = self.convert_coverage_to_xml(coverage_file)
        if not xml_path:
            return 0.0

        coverage_percent = self.parse_coverage_percentage(xml_path)
        self.logger.info(f"Final code coverage: {coverage_percent}%")
        return coverage_percent
