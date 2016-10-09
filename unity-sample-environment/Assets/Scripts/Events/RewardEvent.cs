using UnityEngine;
using System.Collections.Generic;

namespace MLPlayer {
	public class RewardEvent : MonoBehaviour {
        float reward;
		[SerializeField] float rewardDefault;
        [SerializeField] float rewardUpdated;

        void OnEvent(GameObject other) {
			if (other.tag == Defs.PLAYER_TAG) {
                if(other.gameObject.GetComponent<Agent>().state.pushedButton==true)
                {
                    Debug.Log("reward updated");
                    reward = rewardUpdated;
                }
                else
                {
                    Debug.Log("reward default");
                    reward = rewardDefault;
                }
				other.GetComponent<Agent> ().AddReward (reward);
				Debug.Log ("reward:" + reward.ToString ());
				gameObject.SetActive (false);
			}
		}

		void OnTriggerEnter(Collider other) {
			OnEvent (other.gameObject);
		}

		void OnCollisionEnter(Collision collision) {
			OnEvent (collision.gameObject);
		}
	}
}