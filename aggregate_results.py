import glob

def aggregate_results(file_pattern="pm_pinn_seed_success_log.txt"):
    chunk_files = glob.glob(file_pattern)
    
    if not chunk_files:
        print(f"No files found matching '{file_pattern}'.")
        return

    all_success_rates = []
    all_state_dists = []
    all_visual_dists = []

    for file_path in chunk_files:
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num == 80:
                        break
                    line = line.strip()
                    if line:  
                        try:
                            # Split the line by whitespace into separate strings
                            parts = line.split()
                            
                            # Parse the first column (Success Rate)
                            score = float(parts[0])
                            all_success_rates.append(score)
                            
                            # Parse the new columns if they exist
                            if len(parts) >= 3:
                                all_state_dists.append(float(parts[1]))
                                all_visual_dists.append(float(parts[2]))
                                
                        except ValueError:
                            print(f"[Warning] Could not parse line {line_num} in {file_path}: '{line}'")
        except Exception as e:
            print(f"[Error] Failed to read {file_path}: {e}")

    if not all_success_rates:
        print("No valid success rates found in the files.")
        return

    total_seeds = len(all_success_rates)
    avg_success = sum(all_success_rates) / total_seeds
    
    # Safely compute averages for the new metrics
    avg_state_dist = sum(all_state_dists) / len(all_state_dists) if all_state_dists else float('nan')
    avg_visual_dist = sum(all_visual_dists) / len(all_visual_dists) if all_visual_dists else float('nan')

    print("=" * 50)
    print("🎯 PUSH-T SEED EVALUATION REPORT")
    print("=" * 50)
    print(f"Files Processed  : {len(chunk_files)}")
    print(f"Total Seeds Run  : {total_seeds}")
    print("-" * 50)
    print(f"Average Success       : {avg_success:.4f}")
    if all_state_dists:
        print(f"Average State Dist    : {avg_state_dist:.4f}")
    if all_visual_dists:
        print(f"Average Visual Dist   : {avg_visual_dist:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    aggregate_results()