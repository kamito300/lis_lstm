using UnityEngine;
using System.Collections;

namespace MLPlayer
{
    public class reachGoal : MonoBehaviour
    {

        [SerializeField]
        float rewardOnReachGoal;


        bool IsPlayer(GameObject obj)
        {
            return obj.tag == Defs.PLAYER_TAG;
        }

        void OnCollisionEnter(Collision other)
        {
            if (IsPlayer(other.gameObject))
            {
                Debug.Log("Reach The Goal");
                other.gameObject.GetComponent<Agent>().AddReward(rewardOnReachGoal);
                other.gameObject.GetComponent<Agent>().state.reachedEnd = true;
            }
        }
    }
}