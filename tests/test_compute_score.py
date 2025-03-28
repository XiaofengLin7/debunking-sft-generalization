from reil.utils.reward_score.sokoban import compute_score

def test_compute_score():
    ground_truth = 1
    solution_str = "<answer> Up </answer> | <answer> Down </answer> | <answer> Left </answer> | <answer> Right </answer>"
    print(compute_score(solution_str, ground_truth))
    
if __name__ == "__main__":
    test_compute_score()