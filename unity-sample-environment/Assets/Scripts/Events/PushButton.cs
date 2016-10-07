using UnityEngine;
using System.Collections;

namespace MLPlayer
{
    public class PushButton : MonoBehaviour
    {

        [SerializeField]
        float rewardOnPushButton;
        public GameObject door;


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
                door = this.gameObject.transform.parent.gameObject.transform.FindChild("Door").gameObject;
                Vector3 v = door.transform.localPosition;
                v.y = 22;
                door.transform.localPosition = v;
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