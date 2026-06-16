#!/usr/bin/env python3
"""Monitor SageMaker instance and AWS resource usage."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def get_instance_metadata():
    """Get SageMaker instance metadata."""
    try:
        result = subprocess.run(
            ["cat", "/opt/ml/metadata/resource-metadata.json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return {}


def get_cpu_usage():
    """Get current CPU utilization percentage."""
    try:
        result = subprocess.run(
            ["top", "-bn1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "Cpu(s)" in line:
                # Extract idle percentage
                parts = line.split(",")
                for part in parts:
                    if "id" in part:
                        idle = float(part.split()[0])
                        return round(100 - idle, 2)
    except Exception:
        pass
    return None


def get_memory_usage():
    """Get current memory utilization percentage."""
    try:
        result = subprocess.run(
            ["free", "-m"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = result.stdout.split("\n")
        if len(lines) > 1:
            mem_line = lines[1].split()
            total = float(mem_line[1])
            used = float(mem_line[2])
            return round((used / total) * 100, 2)
    except Exception:
        pass
    return None


def get_disk_usage():
    """Get disk usage for shared directory."""
    try:
        result = subprocess.run(
            ["df", "-h", "/home/sagemaker-user/shared"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = result.stdout.split("\n")
        if len(lines) > 1:
            parts = lines[1].split()
            return {
                "filesystem": parts[0],
                "total": parts[1],
                "used": parts[2],
                "available": parts[3],
                "use_percent": parts[4],
                "mount": parts[5] if len(parts) > 5 else "/home/sagemaker-user/shared",
            }
    except Exception:
        pass
    return {}


def monitor_resources(output_path: Path = None):
    """Monitor and display current resource usage."""
    
    metadata = get_instance_metadata()
    cpu = get_cpu_usage()
    memory = get_memory_usage()
    disk = get_disk_usage()
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instance": {
            "type": metadata.get("ResourceName", "unknown"),
            "arn": metadata.get("ResourceArn", "unknown"),
        },
        "cpu_utilization_percent": cpu,
        "memory_utilization_percent": memory,
        "disk_usage": disk,
    }
    
    # Display
    print("=" * 60)
    print("AWS SageMaker Resource Monitor")
    print("=" * 60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Instance: {report['instance']['type']}")
    print(f"CPU: {cpu}%" if cpu else "CPU: N/A")
    print(f"Memory: {memory}%" if memory else "Memory: N/A")
    if disk:
        print(f"Disk: {disk.get('used')} / {disk.get('total')} ({disk.get('use_percent')})")
    print("=" * 60)
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✅ Saved to {output_path}")
    
    return report


if __name__ == "__main__":
    import sys
    
    output = None
    if len(sys.argv) > 1:
        output = Path(sys.argv[1])
    
    monitor_resources(output)
