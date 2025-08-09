"""
Globalization and Internationalization Support for QNet-NO

This module provides comprehensive globalization features including:
- Multi-language support for user interfaces and documentation
- Multi-region deployment configuration
- Compliance with international regulations (GDPR, CCPA, PDPA)
- Currency and unit localization
- Time zone handling for distributed quantum computing

Author: Terry - Terragon Labs
Date: 2025-08-09
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SupportedLocale(Enum):
    """Supported locales for internationalization."""
    EN_US = "en-US"  # English (United States)
    EN_GB = "en-GB"  # English (United Kingdom)
    ES_ES = "es-ES"  # Spanish (Spain)
    ES_MX = "es-MX"  # Spanish (Mexico)
    FR_FR = "fr-FR"  # French (France)
    FR_CA = "fr-CA"  # French (Canada)
    DE_DE = "de-DE"  # German (Germany)
    JA_JP = "ja-JP"  # Japanese (Japan)
    ZH_CN = "zh-CN"  # Chinese (China)
    ZH_TW = "zh-TW"  # Chinese (Taiwan)
    KO_KR = "ko-KR"  # Korean (South Korea)
    IT_IT = "it-IT"  # Italian (Italy)
    PT_BR = "pt-BR"  # Portuguese (Brazil)
    RU_RU = "ru-RU"  # Russian (Russia)
    AR_SA = "ar-SA"  # Arabic (Saudi Arabia)


class SupportedRegion(Enum):
    """Supported regions for deployment and compliance."""
    US_EAST = "us-east-1"      # United States East
    US_WEST = "us-west-2"      # United States West
    EU_WEST = "eu-west-1"      # Europe (Ireland)
    EU_CENTRAL = "eu-central-1" # Europe (Germany)
    ASIA_PACIFIC = "ap-northeast-1"  # Asia Pacific (Tokyo)
    ASIA_SOUTHEAST = "ap-southeast-1" # Asia Pacific (Singapore)
    CANADA = "ca-central-1"    # Canada
    AUSTRALIA = "ap-southeast-2" # Australia
    BRAZIL = "sa-east-1"       # South America (Brazil)
    INDIA = "ap-south-1"       # India
    UK = "eu-west-2"          # United Kingdom
    JAPAN = "ap-northeast-1"   # Japan
    CHINA = "cn-north-1"       # China
    KOREA = "ap-northeast-2"   # South Korea


@dataclass
class RegionConfig:
    """Configuration for a specific deployment region."""
    region: SupportedRegion
    primary_locale: SupportedLocale
    supported_locales: List[SupportedLocale]
    compliance_requirements: List[str]
    currency: str
    timezone: str
    quantum_hardware_available: bool = False
    data_sovereignty_required: bool = False
    
    # Quantum-specific regional settings
    entanglement_distance_limit_km: float = 1000.0  # Max entanglement distance
    quantum_network_latency_ms: float = 50.0        # Expected network latency
    classical_quantum_bandwidth_mbps: float = 100.0  # Classical-quantum bandwidth


class ComplianceFramework:
    """Framework for handling international compliance requirements."""
    
    def __init__(self):
        self.compliance_rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance rules for different regulations."""
        return {
            "GDPR": {
                "name": "General Data Protection Regulation",
                "regions": ["EU_WEST", "EU_CENTRAL", "UK"],
                "requirements": [
                    "data_encryption_at_rest",
                    "data_encryption_in_transit", 
                    "right_to_erasure",
                    "data_portability",
                    "consent_management",
                    "privacy_by_design",
                    "data_processing_logs"
                ],
                "quantum_specific": [
                    "quantum_state_privacy",
                    "entanglement_metadata_protection",
                    "measurement_result_anonymization"
                ]
            },
            "CCPA": {
                "name": "California Consumer Privacy Act",
                "regions": ["US_WEST", "US_EAST"],
                "requirements": [
                    "consumer_privacy_rights",
                    "data_sale_opt_out",
                    "personal_info_disclosure",
                    "data_deletion_requests"
                ],
                "quantum_specific": [
                    "quantum_computation_transparency",
                    "algorithm_explainability"
                ]
            },
            "PDPA": {
                "name": "Personal Data Protection Act",
                "regions": ["ASIA_SOUTHEAST", "ASIA_PACIFIC"],
                "requirements": [
                    "data_breach_notification",
                    "consent_management",
                    "data_portability",
                    "personal_data_protection"
                ],
                "quantum_specific": [
                    "cross_border_quantum_data_transfer",
                    "quantum_key_distribution_compliance"
                ]
            },
            "SOC2": {
                "name": "Service Organization Control 2",
                "regions": ["US_EAST", "US_WEST", "CANADA"],
                "requirements": [
                    "security_controls",
                    "availability_monitoring",
                    "processing_integrity",
                    "confidentiality_measures",
                    "privacy_controls"
                ],
                "quantum_specific": [
                    "quantum_circuit_integrity",
                    "entanglement_security_audit",
                    "quantum_error_rate_monitoring"
                ]
            }
        }
    
    def get_applicable_regulations(self, region: SupportedRegion) -> List[str]:
        """Get applicable regulations for a specific region."""
        applicable = []
        for regulation, details in self.compliance_rules.items():
            if region.value in details["regions"]:
                applicable.append(regulation)
        return applicable
    
    def validate_compliance(self, region: SupportedRegion, 
                          configuration: Dict[str, Any]) -> Dict[str, bool]:
        """Validate compliance for a given region and configuration."""
        results = {}
        regulations = self.get_applicable_regulations(region)
        
        for regulation in regulations:
            requirements = self.compliance_rules[regulation]["requirements"]
            quantum_requirements = self.compliance_rules[regulation].get("quantum_specific", [])
            
            all_requirements = requirements + quantum_requirements
            compliance_score = 0
            
            for requirement in all_requirements:
                if self._check_requirement(requirement, configuration):
                    compliance_score += 1
            
            results[regulation] = compliance_score / len(all_requirements) >= 0.8  # 80% compliance threshold
        
        return results
    
    def _check_requirement(self, requirement: str, config: Dict[str, Any]) -> bool:
        """Check if a specific requirement is met in the configuration."""
        requirement_checks = {
            "data_encryption_at_rest": lambda c: c.get("encryption", {}).get("at_rest", False),
            "data_encryption_in_transit": lambda c: c.get("encryption", {}).get("in_transit", False),
            "consent_management": lambda c: c.get("privacy", {}).get("consent_system", False),
            "data_processing_logs": lambda c: c.get("logging", {}).get("data_processing", False),
            "quantum_state_privacy": lambda c: c.get("quantum", {}).get("state_privacy", False),
            "quantum_circuit_integrity": lambda c: c.get("quantum", {}).get("circuit_verification", False),
            "entanglement_security_audit": lambda c: c.get("quantum", {}).get("entanglement_audit", False)
        }
        
        check_func = requirement_checks.get(requirement, lambda c: True)  # Default to True if unknown
        return check_func(config)


class LocalizationManager:
    """Manager for handling multi-language localization."""
    
    def __init__(self, default_locale: SupportedLocale = SupportedLocale.EN_US):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations = self._load_translations()
        
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation dictionaries for supported locales."""
        # In a real implementation, this would load from files
        return {
            "en-US": {
                "quantum_advantage": "Quantum Advantage",
                "entanglement_quality": "Entanglement Quality",
                "schmidt_rank": "Schmidt Rank",
                "scheduling_complete": "Scheduling Complete",
                "network_nodes": "Network Nodes",
                "task_assignment": "Task Assignment",
                "optimization_running": "Optimization Running...",
                "algorithm_converged": "Algorithm Converged",
                "performance_metrics": "Performance Metrics",
                "quantum_computation": "Quantum Computation",
                "classical_processing": "Classical Processing",
                "hybrid_optimization": "Hybrid Optimization",
                "statistical_significance": "Statistical Significance",
                "experimental_validation": "Experimental Validation"
            },
            "es-ES": {
                "quantum_advantage": "Ventaja Cuántica",
                "entanglement_quality": "Calidad de Entrelazamiento", 
                "schmidt_rank": "Rango de Schmidt",
                "scheduling_complete": "Programación Completa",
                "network_nodes": "Nodos de Red",
                "task_assignment": "Asignación de Tareas",
                "optimization_running": "Optimización en Ejecución...",
                "algorithm_converged": "Algoritmo Convergido",
                "performance_metrics": "Métricas de Rendimiento",
                "quantum_computation": "Computación Cuántica",
                "classical_processing": "Procesamiento Clásico",
                "hybrid_optimization": "Optimización Híbrida",
                "statistical_significance": "Significancia Estadística",
                "experimental_validation": "Validación Experimental"
            },
            "fr-FR": {
                "quantum_advantage": "Avantage Quantique",
                "entanglement_quality": "Qualité d'Intrication",
                "schmidt_rank": "Rang de Schmidt", 
                "scheduling_complete": "Planification Terminée",
                "network_nodes": "Nœuds de Réseau",
                "task_assignment": "Attribution des Tâches",
                "optimization_running": "Optimisation en Cours...",
                "algorithm_converged": "Algorithme Convergé",
                "performance_metrics": "Métriques de Performance",
                "quantum_computation": "Calcul Quantique",
                "classical_processing": "Traitement Classique",
                "hybrid_optimization": "Optimisation Hybride",
                "statistical_significance": "Signification Statistique",
                "experimental_validation": "Validation Expérimentale"
            },
            "de-DE": {
                "quantum_advantage": "Quantenvorteil",
                "entanglement_quality": "Verschränkungsqualität",
                "schmidt_rank": "Schmidt-Rang",
                "scheduling_complete": "Terminplanung Abgeschlossen",
                "network_nodes": "Netzwerkknoten", 
                "task_assignment": "Aufgabenzuweisung",
                "optimization_running": "Optimierung Läuft...",
                "algorithm_converged": "Algorithmus Konvergiert",
                "performance_metrics": "Leistungsmetriken",
                "quantum_computation": "Quantenberechnung",
                "classical_processing": "Klassische Verarbeitung",
                "hybrid_optimization": "Hybride Optimierung",
                "statistical_significance": "Statistische Signifikanz",
                "experimental_validation": "Experimentelle Validierung"
            },
            "ja-JP": {
                "quantum_advantage": "量子優位性",
                "entanglement_quality": "もつれの品質",
                "schmidt_rank": "シュミット・ランク",
                "scheduling_complete": "スケジュール完了", 
                "network_nodes": "ネットワークノード",
                "task_assignment": "タスク割り当て",
                "optimization_running": "最適化実行中...",
                "algorithm_converged": "アルゴリズム収束",
                "performance_metrics": "性能指標",
                "quantum_computation": "量子計算",
                "classical_processing": "古典処理",
                "hybrid_optimization": "ハイブリッド最適化",
                "statistical_significance": "統計的有意性",
                "experimental_validation": "実験的検証"
            },
            "zh-CN": {
                "quantum_advantage": "量子优势",
                "entanglement_quality": "纠缠质量",
                "schmidt_rank": "施密特秩",
                "scheduling_complete": "调度完成",
                "network_nodes": "网络节点",
                "task_assignment": "任务分配", 
                "optimization_running": "优化运行中...",
                "algorithm_converged": "算法收敛",
                "performance_metrics": "性能指标",
                "quantum_computation": "量子计算",
                "classical_processing": "经典处理",
                "hybrid_optimization": "混合优化",
                "statistical_significance": "统计显著性",
                "experimental_validation": "实验验证"
            }
        }
    
    def set_locale(self, locale: SupportedLocale) -> None:
        """Set the current locale."""
        self.current_locale = locale
        logger.info(f"Locale set to {locale.value}")
    
    def translate(self, key: str, locale: Optional[SupportedLocale] = None) -> str:
        """Translate a key to the specified or current locale."""
        target_locale = locale or self.current_locale
        locale_translations = self.translations.get(target_locale.value, {})
        
        # Fallback to default locale if translation not found
        if key not in locale_translations:
            fallback_translations = self.translations.get(self.default_locale.value, {})
            return fallback_translations.get(key, key)  # Return key if no translation found
        
        return locale_translations[key]
    
    def get_supported_locales(self) -> List[SupportedLocale]:
        """Get list of supported locales."""
        return list(SupportedLocale)
    
    def format_number(self, number: float, locale: Optional[SupportedLocale] = None) -> str:
        """Format number according to locale conventions."""
        target_locale = locale or self.current_locale
        
        # Simple formatting based on locale
        if target_locale in [SupportedLocale.EN_US, SupportedLocale.EN_GB]:
            return f"{number:,.2f}"
        elif target_locale in [SupportedLocale.DE_DE, SupportedLocale.FR_FR]:
            # European formatting (comma for decimal, period for thousands)
            return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            return f"{number:.2f}"
    
    def format_percentage(self, percentage: float, locale: Optional[SupportedLocale] = None) -> str:
        """Format percentage according to locale conventions."""
        formatted_number = self.format_number(percentage * 100, locale)
        return f"{formatted_number}%"


class GlobalConfigurationManager:
    """Manager for global configuration across regions and locales."""
    
    def __init__(self):
        self.compliance = ComplianceFramework()
        self.localization = LocalizationManager()
        self.region_configs = self._initialize_region_configs()
        
    def _initialize_region_configs(self) -> Dict[SupportedRegion, RegionConfig]:
        """Initialize configuration for each supported region."""
        configs = {
            SupportedRegion.US_EAST: RegionConfig(
                region=SupportedRegion.US_EAST,
                primary_locale=SupportedLocale.EN_US,
                supported_locales=[SupportedLocale.EN_US, SupportedLocale.ES_MX],
                compliance_requirements=["CCPA", "SOC2"],
                currency="USD",
                timezone="America/New_York",
                quantum_hardware_available=True,
                entanglement_distance_limit_km=2000.0
            ),
            
            SupportedRegion.EU_WEST: RegionConfig(
                region=SupportedRegion.EU_WEST,
                primary_locale=SupportedLocale.EN_GB,
                supported_locales=[SupportedLocale.EN_GB, SupportedLocale.FR_FR, SupportedLocale.DE_DE],
                compliance_requirements=["GDPR"],
                currency="EUR",
                timezone="Europe/Dublin",
                quantum_hardware_available=True,
                data_sovereignty_required=True,
                entanglement_distance_limit_km=1500.0
            ),
            
            SupportedRegion.ASIA_PACIFIC: RegionConfig(
                region=SupportedRegion.ASIA_PACIFIC,
                primary_locale=SupportedLocale.JA_JP,
                supported_locales=[SupportedLocale.JA_JP, SupportedLocale.KO_KR, SupportedLocale.ZH_CN],
                compliance_requirements=["PDPA"],
                currency="JPY",
                timezone="Asia/Tokyo",
                quantum_hardware_available=True,
                entanglement_distance_limit_km=1200.0
            ),
            
            SupportedRegion.CHINA: RegionConfig(
                region=SupportedRegion.CHINA,
                primary_locale=SupportedLocale.ZH_CN,
                supported_locales=[SupportedLocale.ZH_CN],
                compliance_requirements=["PIPL"],  # Personal Information Protection Law
                currency="CNY",
                timezone="Asia/Shanghai",
                quantum_hardware_available=True,
                data_sovereignty_required=True,
                entanglement_distance_limit_km=3000.0  # Larger country
            )
        }
        
        # Add remaining regions with basic configurations
        remaining_regions = set(SupportedRegion) - set(configs.keys())
        for region in remaining_regions:
            configs[region] = self._create_default_region_config(region)
        
        return configs
    
    def _create_default_region_config(self, region: SupportedRegion) -> RegionConfig:
        """Create default configuration for a region."""
        # Simple mapping of region to likely locale
        region_locale_map = {
            SupportedRegion.CANADA: SupportedLocale.EN_US,
            SupportedRegion.BRAZIL: SupportedLocale.PT_BR,
            SupportedRegion.AUSTRALIA: SupportedLocale.EN_US,
            SupportedRegion.INDIA: SupportedLocale.EN_US,
            SupportedRegion.UK: SupportedLocale.EN_GB,
            SupportedRegion.KOREA: SupportedLocale.KO_KR
        }
        
        return RegionConfig(
            region=region,
            primary_locale=region_locale_map.get(region, SupportedLocale.EN_US),
            supported_locales=[region_locale_map.get(region, SupportedLocale.EN_US)],
            compliance_requirements=[],
            currency="USD",  # Default currency
            timezone="UTC",
            quantum_hardware_available=False,  # Conservative default
            entanglement_distance_limit_km=1000.0
        )
    
    def get_region_config(self, region: SupportedRegion) -> RegionConfig:
        """Get configuration for a specific region."""
        return self.region_configs.get(region, self._create_default_region_config(region))
    
    def validate_deployment_config(self, region: SupportedRegion, 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate deployment configuration for a region."""
        region_config = self.get_region_config(region)
        
        # Check compliance
        compliance_results = self.compliance.validate_compliance(region, config)
        
        # Check locale support
        requested_locales = config.get("locales", [region_config.primary_locale.value])
        supported_locale_values = [loc.value for loc in region_config.supported_locales]
        unsupported_locales = [loc for loc in requested_locales if loc not in supported_locale_values]
        
        # Check quantum hardware requirements
        quantum_required = config.get("quantum", {}).get("hardware_required", False)
        quantum_available = region_config.quantum_hardware_available
        
        validation_results = {
            "region": region.value,
            "compliance": compliance_results,
            "compliance_passed": all(compliance_results.values()),
            "locale_support": {
                "requested": requested_locales,
                "supported": supported_locale_values,
                "unsupported": unsupported_locales,
                "all_supported": len(unsupported_locales) == 0
            },
            "quantum_hardware": {
                "required": quantum_required,
                "available": quantum_available,
                "compatible": not quantum_required or quantum_available
            },
            "data_sovereignty": {
                "required": region_config.data_sovereignty_required,
                "configured": config.get("data_sovereignty", {}).get("enabled", False)
            },
            "overall_valid": (
                all(compliance_results.values()) and
                len(unsupported_locales) == 0 and
                (not quantum_required or quantum_available)
            )
        }
        
        return validation_results
    
    def get_deployment_recommendations(self, requirements: Dict[str, Any]) -> List[SupportedRegion]:
        """Get recommended regions based on requirements."""
        suitable_regions = []
        
        for region, config in self.region_configs.items():
            # Check locale requirements
            required_locales = requirements.get("locales", [])
            if required_locales:
                supported_locale_values = [loc.value for loc in config.supported_locales]
                if not all(loc in supported_locale_values for loc in required_locales):
                    continue
            
            # Check quantum hardware requirements
            if requirements.get("quantum_hardware", False) and not config.quantum_hardware_available:
                continue
            
            # Check compliance requirements
            required_compliance = requirements.get("compliance", [])
            available_compliance = self.compliance.get_applicable_regulations(region)
            if not all(comp in available_compliance for comp in required_compliance):
                continue
            
            # Check data sovereignty requirements
            if requirements.get("data_sovereignty", False) and not config.data_sovereignty_required:
                continue
            
            suitable_regions.append(region)
        
        # Sort by preference (simplified - could be more sophisticated)
        preference_order = [
            SupportedRegion.US_EAST, SupportedRegion.EU_WEST, SupportedRegion.ASIA_PACIFIC,
            SupportedRegion.US_WEST, SupportedRegion.EU_CENTRAL, SupportedRegion.CANADA
        ]
        
        return sorted(suitable_regions, key=lambda r: preference_order.index(r) if r in preference_order else 999)


# Global instances
global_config = GlobalConfigurationManager()
localization = LocalizationManager()
compliance = ComplianceFramework()


def get_global_config() -> GlobalConfigurationManager:
    """Get the global configuration manager instance."""
    return global_config


def translate(key: str, locale: Optional[SupportedLocale] = None) -> str:
    """Global translation function."""
    return localization.translate(key, locale)


def set_global_locale(locale: SupportedLocale) -> None:
    """Set the global locale."""
    localization.set_locale(locale)


def validate_region_deployment(region: SupportedRegion, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate deployment configuration for a region."""
    return global_config.validate_deployment_config(region, config)


def get_recommended_regions(requirements: Dict[str, Any]) -> List[SupportedRegion]:
    """Get recommended deployment regions based on requirements."""
    return global_config.get_deployment_recommendations(requirements)


# Example usage and testing functions
def demonstrate_globalization():
    """Demonstrate globalization features."""
    print("🌍 QNet-NO Globalization Features Demonstration")
    print("=" * 60)
    
    # Test translations
    print("\n📖 Multi-Language Support:")
    test_keys = ["quantum_advantage", "entanglement_quality", "optimization_running"]
    test_locales = [SupportedLocale.EN_US, SupportedLocale.ES_ES, SupportedLocale.JA_JP, SupportedLocale.ZH_CN]
    
    for key in test_keys:
        print(f"\n'{key}':")
        for locale in test_locales:
            translation = translate(key, locale)
            print(f"  {locale.value}: {translation}")
    
    # Test region recommendations
    print("\n🗺️ Region Recommendations:")
    
    requirements = {
        "locales": ["en-US", "es-ES"],
        "quantum_hardware": True,
        "compliance": ["GDPR"],
        "data_sovereignty": True
    }
    
    print(f"Requirements: {requirements}")
    recommendations = get_recommended_regions(requirements)
    print(f"Recommended regions: {[r.value for r in recommendations]}")
    
    # Test compliance validation
    print("\n🛡️ Compliance Validation:")
    
    test_config = {
        "encryption": {"at_rest": True, "in_transit": True},
        "privacy": {"consent_system": True},
        "logging": {"data_processing": True},
        "quantum": {"state_privacy": True, "circuit_verification": True}
    }
    
    for region in [SupportedRegion.EU_WEST, SupportedRegion.US_EAST]:
        validation = validate_region_deployment(region, test_config)
        print(f"{region.value}: {'✅ Valid' if validation['overall_valid'] else '❌ Invalid'}")
        print(f"  Compliance: {validation['compliance']}")


if __name__ == "__main__":
    demonstrate_globalization()