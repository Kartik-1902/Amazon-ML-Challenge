#!/usr/bin/env python3
"""
Training Monitor - Check if your ML training is stuck or running
Run this in a SEPARATE terminal while your training is running
"""

import psutil
import time
import os
import sys

def find_python_processes():
    """Find all Python processes"""
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                python_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return python_procs

def monitor_process(duration_seconds=60):
    """Monitor Python processes for activity"""
    
    print("="*70)
    print("🔍 TRAINING PROCESS MONITOR")
    print("="*70)
    print(f"Monitoring for {duration_seconds} seconds...\n")
    
    # Find Python processes
    procs = find_python_processes()
    
    if not procs:
        print("❌ No Python processes found!")
        print("   Make sure your training script is running.")
        return
    
    print(f"Found {len(procs)} Python process(es):\n")
    
    # Display processes
    for proc in procs:
        try:
            cmdline = ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else 'N/A'
            print(f"📌 PID {proc.info['pid']}: {cmdline[:60]}")
        except:
            pass
    
    print("\n" + "-"*70)
    print("⏱️  Monitoring CPU & Memory usage...")
    print("-"*70)
    print(f"{'Time':<10} {'PID':<8} {'CPU %':<10} {'Memory MB':<12} {'Status'}")
    print("-"*70)
    
    # Monitor for specified duration
    samples = duration_seconds // 5  # Sample every 5 seconds
    cpu_history = {proc.info['pid']: [] for proc in procs}
    
    for i in range(samples):
        time.sleep(5)
        
        for proc in procs:
            try:
                cpu = proc.cpu_percent(interval=0.1)
                mem = proc.memory_info().rss / (1024 * 1024)  # Convert to MB
                
                cpu_history[proc.info['pid']].append(cpu)
                
                # Determine status
                if cpu > 50:
                    status = "🟢 ACTIVE"
                elif cpu > 5:
                    status = "🟡 WORKING"
                elif cpu > 0.1:
                    status = "🟠 IDLE"
                else:
                    status = "🔴 STUCK?"
                
                elapsed = (i + 1) * 5
                print(f"{elapsed:<10}s {proc.info['pid']:<8} {cpu:<10.1f} {mem:<12.1f} {status}")
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"{elapsed:<10}s {proc.info['pid']:<8} {'ENDED':<10} {'-':<12} {'⚫ FINISHED'}")
    
    print("\n" + "="*70)
    print("📊 ANALYSIS")
    print("="*70)
    
    for pid, cpu_vals in cpu_history.items():
        if not cpu_vals:
            continue
            
        avg_cpu = sum(cpu_vals) / len(cpu_vals)
        max_cpu = max(cpu_vals)
        recent_cpu = cpu_vals[-3:] if len(cpu_vals) >= 3 else cpu_vals
        recent_avg = sum(recent_cpu) / len(recent_cpu)
        
        print(f"\nPID {pid}:")
        print(f"  Average CPU: {avg_cpu:.1f}%")
        print(f"  Max CPU: {max_cpu:.1f}%")
        print(f"  Recent CPU: {recent_avg:.1f}%")
        
        # Diagnosis
        if recent_avg < 0.5 and avg_cpu < 1:
            print(f"  ⚠️  DIAGNOSIS: Likely STUCK or waiting for I/O")
            print(f"     Recommendation: Check if process is waiting for input or deadlocked")
        elif recent_avg > 50:
            print(f"  ✅ DIAGNOSIS: Actively computing (HEALTHY)")
        elif recent_avg > 5:
            print(f"  🟡 DIAGNOSIS: Working normally (may be in I/O operations)")
        else:
            print(f"  🟠 DIAGNOSIS: Low activity (could be stuck or waiting)")
    
    print("\n" + "="*70)
    print("💡 TIPS:")
    print("="*70)
    print("✅ High CPU (>50%): Model is training - GOOD!")
    print("🟡 Medium CPU (5-50%): Normal I/O operations - OKAY")
    print("🟠 Low CPU (0.5-5%): Might be stuck - CHECK LOGS")
    print("🔴 Zero CPU (<0.5%): Likely stuck - CONSIDER RESTARTING")
    print("\n📝 During CV, expect:")
    print("   - Periodic bursts of high CPU (when fitting models)")
    print("   - Short periods of low CPU (between folds)")
    print("   - Total time: 10-30 minutes depending on data size")
    print("="*70)

def quick_check():
    """Quick one-time check"""
    print("🔍 Quick Process Check\n")
    procs = find_python_processes()
    
    if not procs:
        print("❌ No Python processes found")
        return
    
    for proc in procs:
        try:
            cpu = proc.cpu_percent(interval=1.0)
            mem = proc.memory_info().rss / (1024 * 1024)
            cmdline = ' '.join(proc.info['cmdline'][:2]) if proc.info['cmdline'] else 'N/A'
            
            status = "🟢 ACTIVE" if cpu > 20 else "🟡 LOW" if cpu > 1 else "🔴 IDLE"
            
            print(f"PID {proc.info['pid']}: {status}")
            print(f"  CPU: {cpu:.1f}%")
            print(f"  Memory: {mem:.1f} MB")
            print(f"  Command: {cmdline[:60]}")
            print()
        except:
            pass

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ML TRAINING MONITOR")
    print("="*70)
    print("\nOptions:")
    print("  1. Quick check (1 second)")
    print("  2. Monitor for 60 seconds")
    print("  3. Monitor for 300 seconds (5 minutes)")
    print("  4. Custom duration")
    print()
    
    try:
        choice = input("Select option (1-4) [default: 2]: ").strip()
        
        if choice == "1" or choice == "":
            quick_check()
        elif choice == "2" or not choice:
            monitor_process(60)
        elif choice == "3":
            monitor_process(300)
        elif choice == "4":
            duration = int(input("Enter duration in seconds: "))
            monitor_process(duration)
        else:
            print("Invalid choice, running quick check...")
            quick_check()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Falling back to quick check...")
        quick_check()