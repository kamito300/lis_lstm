using UnityEngine;
using System.Collections.Generic;

namespace MLPlayer {
	public class Environment : MonoBehaviour {

		int itemCount = 10;
		float areaSize = 10;
        public GameObject door;
        public GameObject[] doors;
        [SerializeField] List<GameObject> itemPrefabs;
        [SerializeField]
        float closedDoor;

        // Use this for initialization
        void Start () {
		
		}
		
		// Update is called once per frame
		void Update () {
		
		}

		public void OnReset() {
            if (itemPrefabs.Count > 0)
            {
                foreach (Transform i in transform)
                {
                    Destroy(i.gameObject);
                }
                for (int i = 0; i < itemCount; i++)
                {
                    Vector3 pos = new Vector3(
                        UnityEngine.Random.Range(-areaSize, areaSize),
                        1,
                        UnityEngine.Random.Range(-areaSize, areaSize));
                    Quaternion q = Quaternion.Euler(
                        UnityEngine.Random.Range(0f, 360f),
                        UnityEngine.Random.Range(0f, 360f),
                        UnityEngine.Random.Range(0f, 360f)
                        );

                    pos += transform.position;
                    int itemId = UnityEngine.Random.Range(0, itemPrefabs.Count);
                    GameObject obj = (GameObject)GameObject.Instantiate
                        (itemPrefabs[itemId], pos, Quaternion.identity);
                    obj.transform.parent = transform;
                }
            }

            doors = GameObject.FindGameObjectsWithTag("Door");
            foreach (GameObject door in doors)
            {
                Vector3 v = door.transform.localPosition;
                //v.y = 12;
                v.y = closedDoor;
                door.transform.localPosition = v;
            }
        }
	}
}
