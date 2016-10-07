using UnityEngine;
using System.Collections;

namespace MLPlayer
{
    public class falledFloor : MonoBehaviour
    {
        [SerializeField]
        float rewardOnFloor;

        bool IsPlayer(GameObject obj)
        {
            return obj.tag == Defs.PLAYER_TAG;
        }

        void OnCollisionEnter(Collision other)
        {
            if (IsPlayer(other.gameObject))
            {
                Debug.Log("falled to The Floor");
                other.gameObject.GetComponent<Agent>().AddReward(rewardOnFloor);
                other.gameObject.GetComponent<Agent>().state.reachedEnd = true;
            }
        }
    }
}
