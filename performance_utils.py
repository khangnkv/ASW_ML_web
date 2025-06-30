# Basic performance utilities (simplified)

import time
import functools

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def track_time(self, operation_name: str):
        """Simple decorator to track execution time"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                if operation_name not in self.metrics:
                    self.metrics[operation_name] = []
                self.metrics[operation_name].append(execution_time)
                
                print(f"{operation_name} completed in {execution_time:.2f} seconds")
                return result
            return wrapper
        return decorator
    
    def get_metrics_summary(self):
        """Get summary of tracked metrics"""
        summary = {}
        for operation, times in self.metrics.items():
            summary[operation] = {
                'count': len(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times)
            }
        return summary

# Global performance monitor
perf_monitor = PerformanceMonitor()
