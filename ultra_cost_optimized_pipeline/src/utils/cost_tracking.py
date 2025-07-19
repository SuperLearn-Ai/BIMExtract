"""
Cost Tracking Utility for Ultra-Cost-Optimized Pipeline
Monitors and tracks costs across all pipeline stages
"""

import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import threading


@dataclass
class CostEntry:
    """Individual cost entry"""
    stage: str
    operation: str
    cost: float
    timestamp: datetime
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class CostTracker:
    """Thread-safe cost tracking system"""
    
    def __init__(self):
        self.costs: List[CostEntry] = []
        self.stage_totals: Dict[str, float] = defaultdict(float)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        
        # Cost targets per 1K pages (from config)
        self.targets = {
            "visual_parsing": 0.05,
            "llm_processing": 0.04,
            "vector_storage": 0.03,
            "orchestration": 0.03,
            "total": 0.15
        }
        
        self.start_time = datetime.now()
        
    def add_cost(self, stage: str, cost: float, operation: str = "default", metadata: Dict = None):
        """Add a cost entry"""
        with self.lock:
            entry = CostEntry(
                stage=stage,
                operation=operation,
                cost=cost,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            self.costs.append(entry)
            self.stage_totals[stage] += cost
            self.operation_counts[f"{stage}:{operation}"] += 1
            
            # Log if cost exceeds threshold
            if cost > 0.001:  # $0.001 threshold
                logging.warning(f"High cost detected: {stage}:{operation} = ${cost:.6f}")
    
    def get_stage_cost(self, stage: str) -> float:
        """Get total cost for a stage"""
        with self.lock:
            return self.stage_totals[stage]
    
    def get_total_cost(self) -> float:
        """Get total cost across all stages"""
        with self.lock:
            return sum(self.stage_totals.values())
    
    def get_cost_per_1k_pages(self, page_count: int) -> float:
        """Calculate cost per 1000 pages"""
        if page_count == 0:
            return 0.0
        return (self.get_total_cost() * 1000) / page_count
    
    def is_under_budget(self, page_count: int) -> bool:
        """Check if current costs are under budget"""
        cost_per_1k = self.get_cost_per_1k_pages(page_count)
        return cost_per_1k <= self.targets["total"]
    
    def get_savings_vs_premium(self, page_count: int, premium_cost_per_1k: float = 3.00) -> Dict:
        """Calculate savings compared to premium solution"""
        current_cost_per_1k = self.get_cost_per_1k_pages(page_count)
        savings_per_1k = premium_cost_per_1k - current_cost_per_1k
        savings_percentage = (savings_per_1k / premium_cost_per_1k) * 100
        
        return {
            "current_cost_per_1k": current_cost_per_1k,
            "premium_cost_per_1k": premium_cost_per_1k,
            "savings_per_1k": savings_per_1k,
            "savings_percentage": savings_percentage,
            "roi_multiplier": premium_cost_per_1k / current_cost_per_1k if current_cost_per_1k > 0 else float('inf')
        }
    
    def get_stage_breakdown(self) -> Dict[str, Dict]:
        """Get detailed breakdown by stage"""
        with self.lock:
            breakdown = {}
            
            for stage, total_cost in self.stage_totals.items():
                # Get operations for this stage
                stage_operations = {}
                operation_costs = defaultdict(float)
                
                for entry in self.costs:
                    if entry.stage == stage:
                        operation_costs[entry.operation] += entry.cost
                
                stage_operations = dict(operation_costs)
                
                breakdown[stage] = {
                    "total_cost": total_cost,
                    "target_cost": self.targets.get(stage, 0.0),
                    "over_budget": total_cost > self.targets.get(stage, 0.0),
                    "operations": stage_operations,
                    "operation_count": len(stage_operations)
                }
            
            return breakdown
    
    def get_hourly_costs(self) -> Dict[str, float]:
        """Get costs broken down by hour"""
        with self.lock:
            hourly_costs = defaultdict(float)
            
            for entry in self.costs:
                hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
                hourly_costs[hour_key] += entry.cost
            
            return dict(hourly_costs)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        with self.lock:
            if not self.costs:
                return {}
            
            total_operations = len(self.costs)
            total_cost = self.get_total_cost()
            runtime = (datetime.now() - self.start_time).total_seconds()
            
            return {
                "total_operations": total_operations,
                "total_cost": total_cost,
                "average_cost_per_operation": total_cost / total_operations if total_operations > 0 else 0,
                "operations_per_second": total_operations / runtime if runtime > 0 else 0,
                "cost_per_second": total_cost / runtime if runtime > 0 else 0,
                "runtime_seconds": runtime
            }
    
    def get_report(self) -> Dict:
        """Generate comprehensive cost report"""
        with self.lock:
            total_cost = self.get_total_cost()
            
            report = {
                "summary": {
                    "total_cost": total_cost,
                    "target_cost": self.targets["total"],
                    "under_budget": total_cost <= self.targets["total"],
                    "generated_at": datetime.now().isoformat(),
                    "tracking_duration": str(datetime.now() - self.start_time)
                },
                "stage_breakdown": self.get_stage_breakdown(),
                "performance_metrics": self.get_performance_metrics(),
                "hourly_costs": self.get_hourly_costs(),
                "targets": self.targets
            }
            
            return report
    
    def export_to_json(self, filepath: str):
        """Export cost data to JSON file"""
        report = self.get_report()
        
        # Add raw cost entries
        report["raw_entries"] = [entry.to_dict() for entry in self.costs]
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logging.info(f"Cost report exported to {filepath}")
    
    def reset(self):
        """Reset all cost tracking"""
        with self.lock:
            self.costs.clear()
            self.stage_totals.clear()
            self.operation_counts.clear()
            self.start_time = datetime.now()
        
        logging.info("Cost tracker reset")
    
    def set_targets(self, targets: Dict[str, float]):
        """Update cost targets"""
        with self.lock:
            self.targets.update(targets)
        
        logging.info(f"Cost targets updated: {targets}")


class RealTimeCostMonitor:
    """Real-time cost monitoring with alerts"""
    
    def __init__(self, cost_tracker: CostTracker, alert_threshold: float = 0.20):
        self.cost_tracker = cost_tracker
        self.alert_threshold = alert_threshold  # $0.20 per 1K pages
        self.alerts_sent = []
        
    def check_cost_alerts(self, page_count: int) -> List[str]:
        """Check for cost alerts"""
        alerts = []
        
        cost_per_1k = self.cost_tracker.get_cost_per_1k_pages(page_count)
        
        # Overall cost alert
        if cost_per_1k > self.alert_threshold:
            alerts.append(f"COST ALERT: ${cost_per_1k:.4f} per 1K pages exceeds threshold of ${self.alert_threshold}")
        
        # Stage-specific alerts
        breakdown = self.cost_tracker.get_stage_breakdown()
        for stage, data in breakdown.items():
            if data["over_budget"]:
                alerts.append(f"STAGE ALERT: {stage} costs ${data['total_cost']:.6f} exceed target ${data['target_cost']:.6f}")
        
        # Log new alerts
        new_alerts = [alert for alert in alerts if alert not in self.alerts_sent]
        for alert in new_alerts:
            logging.warning(alert)
            self.alerts_sent.append(alert)
        
        return alerts
    
    def get_cost_trend(self, window_minutes: int = 60) -> Dict:
        """Analyze cost trend over time window"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        recent_costs = [
            entry for entry in self.cost_tracker.costs
            if entry.timestamp >= cutoff_time
        ]
        
        if len(recent_costs) < 2:
            return {"trend": "insufficient_data"}
        
        # Simple trend analysis
        costs_by_minute = defaultdict(float)
        for entry in recent_costs:
            minute_key = entry.timestamp.strftime("%Y-%m-%d %H:%M")
            costs_by_minute[minute_key] += entry.cost
        
        cost_values = list(costs_by_minute.values())
        
        # Calculate trend
        if len(cost_values) >= 2:
            recent_avg = sum(cost_values[-5:]) / min(5, len(cost_values))
            earlier_avg = sum(cost_values[:-5]) / max(1, len(cost_values) - 5)
            
            if recent_avg > earlier_avg * 1.2:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_average": recent_avg,
            "window_total": sum(cost_values),
            "data_points": len(cost_values)
        }


# Usage example
def main():
    """Example usage of cost tracking"""
    tracker = CostTracker()
    monitor = RealTimeCostMonitor(tracker)
    
    # Simulate some costs
    tracker.add_cost("visual_parsing", 0.00005, "ocr")
    tracker.add_cost("visual_parsing", 0.00001, "nougat")
    tracker.add_cost("llm_processing", 0.00002, "chunking")
    tracker.add_cost("vector_storage", 0.000001, "embedding")
    
    # Generate report
    report = tracker.get_report()
    print(json.dumps(report, indent=2, default=str))
    
    # Check alerts
    alerts = monitor.check_cost_alerts(page_count=100)
    print("Alerts:", alerts)
    
    # Export report
    tracker.export_to_json("cost_report.json")


if __name__ == "__main__":
    main()