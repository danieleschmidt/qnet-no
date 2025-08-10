#!/usr/bin/env python3
"""
Security Scanner for QNet-NO Quantum Computing Library

Performs comprehensive security analysis to ensure the library is safe
for production use and free from vulnerabilities.

Security Checks:
- Code injection vulnerabilities
- Unsafe imports and eval usage
- Hardcoded secrets and credentials
- File system access controls
- Network security patterns
- Input validation and sanitization
- Quantum-specific security considerations

Author: Terry - Terragon Labs
Date: 2025-08-10
"""

import os
import re
import ast
import sys
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass
from enum import Enum, auto
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class SecuritySeverity(Enum):
    """Security issue severity levels."""
    INFO = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class SecurityCategory(Enum):
    """Security issue categories."""
    CODE_INJECTION = auto()
    UNSAFE_IMPORTS = auto()
    HARDCODED_SECRETS = auto()
    FILE_SYSTEM_ACCESS = auto()
    NETWORK_SECURITY = auto()
    INPUT_VALIDATION = auto()
    CRYPTOGRAPHIC = auto()
    QUANTUM_SPECIFIC = auto()
    DEPENDENCY_VULNERABILITY = auto()


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    category: SecurityCategory
    severity: SecuritySeverity
    file_path: str
    line_number: int
    description: str
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID


class SecurityScanner:
    """
    Comprehensive security scanner for quantum computing libraries.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues: List[SecurityIssue] = []
        self.scanned_files: Set[str] = set()
        
        # Security patterns to detect
        self.security_patterns = self._initialize_security_patterns()
        
        # Known safe patterns (to reduce false positives)
        self.safe_patterns = self._initialize_safe_patterns()
        
        logger.info(f"Initialized security scanner for {self.project_root}")
    
    def _initialize_security_patterns(self) -> Dict[SecurityCategory, List[Dict[str, Any]]]:
        """Initialize security vulnerability patterns."""
        
        return {
            SecurityCategory.CODE_INJECTION: [
                {
                    'pattern': r'eval\s*\(',
                    'severity': SecuritySeverity.CRITICAL,
                    'description': 'Use of eval() function can lead to code injection',
                    'recommendation': 'Use ast.literal_eval() or safer alternatives',
                    'cwe_id': 'CWE-94'
                },
                {
                    'pattern': r'exec\s*\(',
                    'severity': SecuritySeverity.CRITICAL,
                    'description': 'Use of exec() function can lead to code injection',
                    'recommendation': 'Use safer alternatives or strict input validation',
                    'cwe_id': 'CWE-94'
                },
                {
                    'pattern': r'compile\s*\(',
                    'severity': SecuritySeverity.HIGH,
                    'description': 'Dynamic code compilation should be carefully reviewed',
                    'recommendation': 'Ensure input is properly validated',
                    'cwe_id': 'CWE-94'
                },
                {
                    'pattern': r'__import__\s*\(',
                    'severity': SecuritySeverity.MEDIUM,
                    'description': 'Dynamic imports should be controlled',
                    'recommendation': 'Use importlib with validation',
                    'cwe_id': 'CWE-470'
                }
            ],
            
            SecurityCategory.HARDCODED_SECRETS: [
                {
                    'pattern': r'password\s*=\s*["\'][^"\']+["\']',
                    'severity': SecuritySeverity.HIGH,
                    'description': 'Hardcoded password detected',
                    'recommendation': 'Use environment variables or secure config',
                    'cwe_id': 'CWE-798'
                },
                {
                    'pattern': r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
                    'severity': SecuritySeverity.HIGH,
                    'description': 'Hardcoded API key detected',
                    'recommendation': 'Use environment variables or secure vault',
                    'cwe_id': 'CWE-798'
                },
                {
                    'pattern': r'secret\s*=\s*["\'][^"\']+["\']',
                    'severity': SecuritySeverity.HIGH,
                    'description': 'Hardcoded secret detected',
                    'recommendation': 'Use secure configuration management',
                    'cwe_id': 'CWE-798'
                },
                {
                    'pattern': r'token\s*=\s*["\'][A-Za-z0-9+/]{20,}["\']',
                    'severity': SecuritySeverity.HIGH,
                    'description': 'Hardcoded token detected',
                    'recommendation': 'Use environment variables',
                    'cwe_id': 'CWE-798'
                }
            ],
            
            SecurityCategory.FILE_SYSTEM_ACCESS: [
                {
                    'pattern': r'open\s*\([^)]*["\']\/[^"\']*["\']',
                    'severity': SecuritySeverity.MEDIUM,
                    'description': 'Absolute path usage may be insecure',
                    'recommendation': 'Use relative paths or validate input paths',
                    'cwe_id': 'CWE-22'
                },
                {
                    'pattern': r'\.\.\/|\.\.\\\|\.\.\\\\',
                    'severity': SecuritySeverity.HIGH,
                    'description': 'Path traversal pattern detected',
                    'recommendation': 'Validate and sanitize file paths',
                    'cwe_id': 'CWE-22'
                },
                {
                    'pattern': r'os\.system\s*\(',
                    'severity': SecuritySeverity.CRITICAL,
                    'description': 'Use of os.system() can lead to command injection',
                    'recommendation': 'Use subprocess with shell=False',
                    'cwe_id': 'CWE-78'
                },
                {
                    'pattern': r'subprocess\.[^(]+\([^)]*shell\s*=\s*True',
                    'severity': SecuritySeverity.HIGH,
                    'description': 'Subprocess with shell=True is dangerous',
                    'recommendation': 'Use shell=False and validate input',
                    'cwe_id': 'CWE-78'
                }
            ],
            
            SecurityCategory.UNSAFE_IMPORTS: [
                {
                    'pattern': r'from\s+__future__\s+import\s+.*',
                    'severity': SecuritySeverity.INFO,
                    'description': 'Future imports detected (informational)',
                    'recommendation': 'Ensure compatibility',
                    'cwe_id': None
                },
                {
                    'pattern': r'import\s+pickle',
                    'severity': SecuritySeverity.MEDIUM,
                    'description': 'Pickle usage can be unsafe with untrusted data',
                    'recommendation': 'Use json or safer serialization methods',
                    'cwe_id': 'CWE-502'
                }
            ],
            
            SecurityCategory.NETWORK_SECURITY: [
                {
                    'pattern': r'urllib\.request\.urlopen\([^)]*["\']http:\/\/[^"\']*["\']',
                    'severity': SecuritySeverity.MEDIUM,
                    'description': 'HTTP (not HTTPS) connection detected',
                    'recommendation': 'Use HTTPS for secure communication',
                    'cwe_id': 'CWE-319'
                },
                {
                    'pattern': r'ssl_context\s*=\s*None',
                    'severity': SecuritySeverity.HIGH,
                    'description': 'SSL context disabled',
                    'recommendation': 'Use proper SSL context validation',
                    'cwe_id': 'CWE-295'
                },
                {
                    'pattern': r'verify\s*=\s*False',
                    'severity': SecuritySeverity.HIGH,
                    'description': 'SSL verification disabled',
                    'recommendation': 'Enable SSL certificate verification',
                    'cwe_id': 'CWE-295'
                }
            ],
            
            SecurityCategory.QUANTUM_SPECIFIC: [
                {
                    'pattern': r'quantum[_-]?key|entanglement[_-]?key',
                    'severity': SecuritySeverity.HIGH,
                    'description': 'Quantum cryptographic key handling detected',
                    'recommendation': 'Ensure quantum keys are properly protected',
                    'cwe_id': 'CWE-320'
                },
                {
                    'pattern': r'fidelity\s*=\s*[01]\.0+',
                    'severity': SecuritySeverity.LOW,
                    'description': 'Perfect fidelity assumption may be unrealistic',
                    'recommendation': 'Use realistic fidelity values for security',
                    'cwe_id': None
                },
                {
                    'pattern': r'noise[_-]?free|noiseless',
                    'severity': SecuritySeverity.INFO,
                    'description': 'Noise-free assumption detected',
                    'recommendation': 'Consider quantum noise in security analysis',
                    'cwe_id': None
                }
            ],
            
            SecurityCategory.INPUT_VALIDATION: [
                {
                    'pattern': r'input\s*\(',
                    'severity': SecuritySeverity.MEDIUM,
                    'description': 'Use of input() function without validation',
                    'recommendation': 'Validate and sanitize user input',
                    'cwe_id': 'CWE-20'
                },
                {
                    'pattern': r'raw_input\s*\(',
                    'severity': SecuritySeverity.MEDIUM,
                    'description': 'Use of raw_input() without validation',
                    'recommendation': 'Validate and sanitize user input',
                    'cwe_id': 'CWE-20'
                }
            ],
            
            SecurityCategory.CRYPTOGRAPHIC: [
                {
                    'pattern': r'md5\s*\(',
                    'severity': SecuritySeverity.MEDIUM,
                    'description': 'MD5 is cryptographically weak',
                    'recommendation': 'Use SHA-256 or stronger hash functions',
                    'cwe_id': 'CWE-327'
                },
                {
                    'pattern': r'sha1\s*\(',
                    'severity': SecuritySeverity.MEDIUM,
                    'description': 'SHA-1 is cryptographically weak',
                    'recommendation': 'Use SHA-256 or stronger hash functions',
                    'cwe_id': 'CWE-327'
                },
                {
                    'pattern': r'random\.seed\s*\([^)]*\)',
                    'severity': SecuritySeverity.LOW,
                    'description': 'Random seed usage detected',
                    'recommendation': 'Ensure seed is not predictable for security',
                    'cwe_id': 'CWE-330'
                }
            ]
        }
    
    def _initialize_safe_patterns(self) -> List[str]:
        """Initialize patterns that are known to be safe."""
        
        return [
            r'# Test.*',  # Test comments
            r'""".*"""',  # Docstrings
            r'logger\..*',  # Logging calls
            r'assert.*',  # Assertions
            r'print\s*\(',  # Print statements (generally safe in this context)
        ]
    
    def scan_project(self) -> Dict[str, Any]:
        """
        Perform comprehensive security scan of the project.
        
        Returns:
            Security scan report
        """
        
        logger.info("Starting comprehensive security scan")
        scan_start_time = time.time()
        
        # Reset scan state
        self.issues = []
        self.scanned_files = set()
        
        # Scan Python files
        python_files = self._find_python_files()
        logger.info(f"Found {len(python_files)} Python files to scan")
        
        for file_path in python_files:
            self._scan_file(file_path)
        
        # Scan configuration files
        config_files = self._find_config_files()
        logger.info(f"Found {len(config_files)} configuration files to scan")
        
        for file_path in config_files:
            self._scan_config_file(file_path)
        
        # Perform dependency vulnerability scan
        self._scan_dependencies()
        
        # Generate report
        scan_time = time.time() - scan_start_time
        report = self._generate_security_report(scan_time)
        
        logger.info(f"Security scan completed in {scan_time:.2f}s")
        logger.info(f"Found {len(self.issues)} security issues")
        
        return report
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        
        python_files = []
        
        # Search patterns
        patterns = ['**/*.py']
        
        # Excluded directories
        excluded_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}
        
        for pattern in patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    # Check if file is in excluded directory
                    if any(excluded_dir in file_path.parts for excluded_dir in excluded_dirs):
                        continue
                    
                    python_files.append(file_path)
        
        return python_files
    
    def _find_config_files(self) -> List[Path]:
        """Find configuration files that may contain sensitive data."""
        
        config_files = []
        
        # Configuration file patterns
        config_patterns = [
            '**/*.json', '**/*.yaml', '**/*.yml', '**/*.ini', '**/*.cfg',
            '**/*.env', '**/.*env*', '**/*.config', '**/*.conf'
        ]
        
        for pattern in config_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    config_files.append(file_path)
        
        return config_files
    
    def _scan_file(self, file_path: Path) -> None:
        """Scan a single Python file for security issues."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Perform line-by-line pattern matching
            for line_num, line in enumerate(lines, 1):
                self._scan_line(file_path, line_num, line)
            
            # Perform AST-based analysis
            self._scan_ast(file_path, content)
            
            self.scanned_files.add(str(file_path))
            
        except Exception as e:
            logger.warning(f"Could not scan file {file_path}: {e}")
    
    def _scan_line(self, file_path: Path, line_number: int, line: str) -> None:
        """Scan a single line for security patterns."""
        
        # Skip if line matches safe patterns
        if self._is_safe_line(line):
            return
        
        # Check each security category
        for category, patterns in self.security_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                
                matches = re.finditer(pattern, line, re.IGNORECASE)
                
                for match in matches:
                    issue = SecurityIssue(
                        category=category,
                        severity=pattern_info['severity'],
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_number,
                        description=pattern_info['description'],
                        code_snippet=line.strip(),
                        recommendation=pattern_info['recommendation'],
                        cwe_id=pattern_info.get('cwe_id')
                    )
                    
                    self.issues.append(issue)
    
    def _is_safe_line(self, line: str) -> bool:
        """Check if line matches known safe patterns."""
        
        line_stripped = line.strip()
        
        # Empty lines or comments
        if not line_stripped or line_stripped.startswith('#'):
            return True
        
        # Check safe patterns
        for safe_pattern in self.safe_patterns:
            if re.match(safe_pattern, line_stripped):
                return True
        
        return False
    
    def _scan_ast(self, file_path: Path, content: str) -> None:
        """Perform AST-based security analysis."""
        
        try:
            tree = ast.parse(content)
            visitor = SecurityASTVisitor(file_path, self.project_root)
            visitor.visit(tree)
            
            # Add issues found by AST visitor
            self.issues.extend(visitor.issues)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"AST analysis failed for {file_path}: {e}")
    
    def _scan_config_file(self, file_path: Path) -> None:
        """Scan configuration files for sensitive data."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Look for potential secrets in config files
            secret_patterns = [
                r'password\s*[:=]\s*.+',
                r'api[_-]?key\s*[:=]\s*.+',
                r'secret\s*[:=]\s*.+',
                r'token\s*[:=]\s*.+'
            ]
            
            for line_num, line in enumerate(lines, 1):
                for pattern in secret_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issue = SecurityIssue(
                            category=SecurityCategory.HARDCODED_SECRETS,
                            severity=SecuritySeverity.HIGH,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            description=f'Potential secret in configuration file',
                            code_snippet=line.strip(),
                            recommendation='Use environment variables or secure vault',
                            cwe_id='CWE-798'
                        )
                        
                        self.issues.append(issue)
            
        except Exception as e:
            logger.warning(f"Could not scan config file {file_path}: {e}")
    
    def _scan_dependencies(self) -> None:
        """Scan project dependencies for known vulnerabilities."""
        
        # Look for requirements files
        req_files = list(self.project_root.glob('**/requirements*.txt'))
        req_files.extend(list(self.project_root.glob('**/pyproject.toml')))
        
        if not req_files:
            logger.info("No dependency files found")
            return
        
        logger.info(f"Scanning {len(req_files)} dependency files")
        
        # For each requirements file, check for known vulnerable packages
        vulnerable_packages = self._get_known_vulnerable_packages()
        
        for req_file in req_files:
            try:
                with open(req_file, 'r') as f:
                    content = f.read()
                
                # Extract package names (simplified)
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Extract package name
                    package_name = line.split('>=')[0].split('==')[0].split('<=')[0].split('>')[0].split('<')[0].split('!')[0]
                    package_name = package_name.strip()
                    
                    if package_name in vulnerable_packages:
                        issue = SecurityIssue(
                            category=SecurityCategory.DEPENDENCY_VULNERABILITY,
                            severity=SecuritySeverity.MEDIUM,
                            file_path=str(req_file.relative_to(self.project_root)),
                            line_number=line_num,
                            description=f'Package {package_name} has known vulnerabilities',
                            code_snippet=line,
                            recommendation='Update to latest secure version',
                            cwe_id='CWE-1104'
                        )
                        
                        self.issues.append(issue)
            
            except Exception as e:
                logger.warning(f"Could not scan dependency file {req_file}: {e}")
    
    def _get_known_vulnerable_packages(self) -> Set[str]:
        """Get list of packages with known vulnerabilities."""
        
        # Simplified list of packages that have had vulnerabilities
        # In production, this would query a vulnerability database
        return {
            'pillow', 'requests', 'urllib3', 'pyyaml', 'jinja2',
            'werkzeug', 'flask', 'django', 'tensorflow', 'numpy'
        }
    
    def _generate_security_report(self, scan_time: float) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        
        # Categorize issues by severity
        severity_counts = {severity: 0 for severity in SecuritySeverity}
        for issue in self.issues:
            severity_counts[issue.severity] += 1
        
        # Categorize issues by category
        category_counts = {category: 0 for category in SecurityCategory}
        for issue in self.issues:
            category_counts[issue.category] += 1
        
        # Calculate risk score
        risk_score = self._calculate_risk_score()
        
        # Determine overall security status
        security_status = self._determine_security_status(severity_counts)
        
        report = {
            'scan_summary': {
                'scan_time': scan_time,
                'files_scanned': len(self.scanned_files),
                'total_issues': len(self.issues),
                'security_status': security_status,
                'risk_score': risk_score
            },
            'severity_breakdown': {
                severity.name: count for severity, count in severity_counts.items()
            },
            'category_breakdown': {
                category.name: count for category, count in category_counts.items()
            },
            'critical_issues': [
                self._issue_to_dict(issue) for issue in self.issues
                if issue.severity == SecuritySeverity.CRITICAL
            ],
            'high_issues': [
                self._issue_to_dict(issue) for issue in self.issues
                if issue.severity == SecuritySeverity.HIGH
            ],
            'all_issues': [
                self._issue_to_dict(issue) for issue in self.issues
            ],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score (0-100)."""
        
        if not self.issues:
            return 0.0
        
        # Weight by severity
        severity_weights = {
            SecuritySeverity.INFO: 1,
            SecuritySeverity.LOW: 2,
            SecuritySeverity.MEDIUM: 5,
            SecuritySeverity.HIGH: 10,
            SecuritySeverity.CRITICAL: 20
        }
        
        total_weight = sum(severity_weights[issue.severity] for issue in self.issues)
        max_possible_weight = len(self.issues) * severity_weights[SecuritySeverity.CRITICAL]
        
        if max_possible_weight == 0:
            return 0.0
        
        risk_score = (total_weight / max_possible_weight) * 100
        return min(100.0, risk_score)
    
    def _determine_security_status(self, severity_counts: Dict[SecuritySeverity, int]) -> str:
        """Determine overall security status."""
        
        if severity_counts[SecuritySeverity.CRITICAL] > 0:
            return "CRITICAL"
        elif severity_counts[SecuritySeverity.HIGH] > 0:
            return "HIGH_RISK"
        elif severity_counts[SecuritySeverity.MEDIUM] > 3:
            return "MEDIUM_RISK"
        elif sum(severity_counts.values()) > 10:
            return "LOW_RISK"
        else:
            return "SECURE"
    
    def _issue_to_dict(self, issue: SecurityIssue) -> Dict[str, Any]:
        """Convert SecurityIssue to dictionary."""
        
        return {
            'category': issue.category.name,
            'severity': issue.severity.name,
            'file_path': issue.file_path,
            'line_number': issue.line_number,
            'description': issue.description,
            'code_snippet': issue.code_snippet,
            'recommendation': issue.recommendation,
            'cwe_id': issue.cwe_id
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        
        recommendations = []
        
        severity_counts = {severity: 0 for severity in SecuritySeverity}
        for issue in self.issues:
            severity_counts[issue.severity] += 1
        
        if severity_counts[SecuritySeverity.CRITICAL] > 0:
            recommendations.append("URGENT: Address all critical security issues immediately")
        
        if severity_counts[SecuritySeverity.HIGH] > 0:
            recommendations.append("Address high-severity issues before production deployment")
        
        # Category-specific recommendations
        category_counts = {category: 0 for category in SecurityCategory}
        for issue in self.issues:
            category_counts[issue.category] += 1
        
        if category_counts[SecurityCategory.CODE_INJECTION] > 0:
            recommendations.append("Implement input validation to prevent code injection")
        
        if category_counts[SecurityCategory.HARDCODED_SECRETS] > 0:
            recommendations.append("Move all secrets to environment variables or secure vault")
        
        if category_counts[SecurityCategory.FILE_SYSTEM_ACCESS] > 0:
            recommendations.append("Review file system access patterns for security")
        
        if category_counts[SecurityCategory.NETWORK_SECURITY] > 0:
            recommendations.append("Ensure all network communications use HTTPS/TLS")
        
        if category_counts[SecurityCategory.QUANTUM_SPECIFIC] > 0:
            recommendations.append("Review quantum cryptographic implementations for security")
        
        # General recommendations
        recommendations.extend([
            "Implement comprehensive input validation throughout the application",
            "Use security linters in CI/CD pipeline",
            "Perform regular dependency vulnerability scans",
            "Consider implementing security monitoring and alerting",
            "Conduct regular security code reviews"
        ])
        
        return recommendations


class SecurityASTVisitor(ast.NodeVisitor):
    """AST visitor for advanced security analysis."""
    
    def __init__(self, file_path: Path, project_root: Path):
        self.file_path = file_path
        self.project_root = project_root
        self.issues: List[SecurityIssue] = []
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes."""
        
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            if func_name in ['eval', 'exec']:
                issue = SecurityIssue(
                    category=SecurityCategory.CODE_INJECTION,
                    severity=SecuritySeverity.CRITICAL,
                    file_path=str(self.file_path.relative_to(self.project_root)),
                    line_number=getattr(node, 'lineno', 0),
                    description=f'Dangerous function {func_name}() detected',
                    code_snippet=f'{func_name}(...)',
                    recommendation='Use safer alternatives',
                    cwe_id='CWE-94'
                )
                self.issues.append(issue)
        
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        
        for alias in node.names:
            if alias.name == 'pickle':
                issue = SecurityIssue(
                    category=SecurityCategory.UNSAFE_IMPORTS,
                    severity=SecuritySeverity.MEDIUM,
                    file_path=str(self.file_path.relative_to(self.project_root)),
                    line_number=getattr(node, 'lineno', 0),
                    description='Pickle import detected - unsafe with untrusted data',
                    code_snippet=f'import {alias.name}',
                    recommendation='Use json or other safe serialization',
                    cwe_id='CWE-502'
                )
                self.issues.append(issue)
        
        self.generic_visit(node)


def run_security_scan(project_root: str = ".") -> Dict[str, Any]:
    """
    Run comprehensive security scan on the project.
    
    Args:
        project_root: Root directory of the project to scan
        
    Returns:
        Security scan report
    """
    
    import time
    
    scanner = SecurityScanner(project_root)
    return scanner.scan_project()


def print_security_report(report: Dict[str, Any]) -> None:
    """Print formatted security report to console."""
    
    print("\n" + "="*80)
    print("SECURITY SCAN REPORT")
    print("="*80)
    
    summary = report['scan_summary']
    print(f"\nScan Summary:")
    print(f"  Files Scanned: {summary['files_scanned']}")
    print(f"  Scan Time: {summary['scan_time']:.2f}s")
    print(f"  Security Status: {summary['security_status']}")
    print(f"  Risk Score: {summary['risk_score']:.1f}/100")
    print(f"  Total Issues: {summary['total_issues']}")
    
    # Severity breakdown
    severity_breakdown = report['severity_breakdown']
    print(f"\nSeverity Breakdown:")
    for severity, count in severity_breakdown.items():
        if count > 0:
            print(f"  {severity}: {count}")
    
    # Critical issues
    critical_issues = report['critical_issues']
    if critical_issues:
        print(f"\nCRITICAL ISSUES ({len(critical_issues)}):")
        for issue in critical_issues:
            print(f"  - {issue['description']}")
            print(f"    File: {issue['file_path']}:{issue['line_number']}")
            print(f"    Code: {issue['code_snippet']}")
            print(f"    Fix: {issue['recommendation']}")
            print()
    
    # High issues
    high_issues = report['high_issues']
    if high_issues:
        print(f"\nHIGH PRIORITY ISSUES ({len(high_issues)}):")
        for issue in high_issues[:5]:  # Show first 5
            print(f"  - {issue['description']}")
            print(f"    File: {issue['file_path']}:{issue['line_number']}")
            print(f"    Fix: {issue['recommendation']}")
            print()
        
        if len(high_issues) > 5:
            print(f"    ... and {len(high_issues) - 5} more high-priority issues")
    
    # Recommendations
    recommendations = report['recommendations']
    print(f"\nRECOMMENDations:")
    for i, rec in enumerate(recommendations[:10], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    # Run security scan when executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description='Security Scanner for QNet-NO')
    parser.add_argument('--project-root', default='.', 
                       help='Root directory of project to scan')
    parser.add_argument('--output-file', 
                       help='Output file for detailed report (JSON)')
    
    args = parser.parse_args()
    
    # Run scan
    report = run_security_scan(args.project_root)
    
    # Print report
    print_security_report(report)
    
    # Save detailed report if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {args.output_file}")
    
    # Exit with appropriate code
    security_status = report['scan_summary']['security_status']
    if security_status in ['CRITICAL', 'HIGH_RISK']:
        sys.exit(1)
    else:
        sys.exit(0)