using UnityEngine;
using System.Collections.Generic;

namespace MLPlayer
{
    public class updateFood : MonoBehaviour
    {
        [SerializeField]
        float rewardOnPushButton;
        [SerializeField]
        List<GameObject> itemPrefabs;

        bool IsPlayer(GameObject obj)
        {
            return obj.tag == Defs.PLAYER_TAG;
        }

        void OnCollisionEnter(Collision other)
        {
            if (IsPlayer(other.gameObject))
            {
                Debug.Log("Push The Button");
                other.gameObject.GetComponent<Agent>().AddReward(rewardOnPushButton);
                this.gameObject.GetComponent<Renderer>().materials[0].color = Color.black;
                // door = this.gameObject.transform.FindChild("Door").gameObject;
                // door = this.gameObject.transform.parent.gameObject.transform.FindChild("Door").gameObject;
                other.gameObject.GetComponent<Agent>().state.pushedButton = true;
                //itemPrefabs = GameObject.Find("PlusRewardItem");
                Debug.Log(other.gameObject.GetComponent<Agent>().state.pushedButton);
            }
        }

        void OnCollisionExit(Collision other)
        {
            if (IsPlayer(other.gameObject))
            {
                Debug.Log("Leave The Button");
                this.gameObject.GetComponent<Renderer>().materials[0].color = new Color(240, 228, 71);
            }
        }
    }
}