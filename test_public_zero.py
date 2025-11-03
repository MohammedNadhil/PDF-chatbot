#!/usr/bin/env python3
"""
Test script to verify public=0 behavior - should strictly return "not enough information" message
"""

import requests
import json

def test_public_zero_behavior():
    """Test that public=0 strictly returns 'not enough information' when no relevant content"""
    
    base_url = "http://localhost:8080"
    
    # Test cases that should return "not enough information" for public=0
    test_cases = [
        {
            "name": "JavaScript Question with Maths Document (public=0)",
            "data": {
                "question": "what is javascript?",
                "source_id": "2b7c65f4a2cb211427b6652055b1c52e",  # maths document
                "public": 0
            },
            "expected_answer": "I don't have enough information in the provided documents to answer this question accurately.",
            "expected_source": "system"
        },
        {
            "name": "PHP Question with Maths Document (public=0)",
            "data": {
                "question": "what is php?",
                "source_id": "2b7c65f4a2cb211427b6652055b1c52e",  # maths document
                "public": 0
            },
            "expected_answer": "I don't have enough information in the provided documents to answer this question accurately.",
            "expected_source": "system"
        },
        {
            "name": "Cooking Question with Education Document (public=0)",
            "data": {
                "question": "how do I cook pasta?",
                "source_id": "preloaded",  # education document
                "public": 0
            },
            "expected_answer": "I don't have enough information in the provided documents to answer this question accurately.",
            "expected_source": "system"
        },
        {
            "name": "Default Behavior (no public param) - should be public=1",
            "data": {
                "question": "what is javascript?",
                "source_id": "2b7c65f4a2cb211427b6652055b1c52e"  # no public param
            },
            "expected_source": "general",  # Should fallback to OpenAI
            "expected_answer": "javascript"  # Should contain javascript info
        }
    ]
    
    print("Testing Public=0 Behavior - Strict 'Not Enough Information' Response")
    print("=" * 70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 50)
        print(f"Question: {test_case['data']['question']}")
        print(f"Source ID: {test_case['data']['source_id']}")
        print(f"Public: {test_case['data'].get('public', 'default (1)')}")
        
        try:
            response = requests.post(
                f"{base_url}/ask",
                json=test_case['data'],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', '')
                source = result.get('source', '')
                
                print(f"✅ Status: SUCCESS")
                print(f"Source: {source}")
                print(f"Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                
                # Check if response matches expectations
                if test_case['expected_source'] in source:
                    print(f"✅ Source check: PASSED")
                else:
                    print(f"❌ Source check: FAILED (expected {test_case['expected_source']}, got {source})")
                
                if 'expected_answer' in test_case:
                    if test_case['expected_answer'].lower() in answer.lower():
                        print(f"✅ Answer check: PASSED")
                    else:
                        print(f"❌ Answer check: FAILED (expected to contain '{test_case['expected_answer']}')")
                else:
                    if test_case['expected_answer'].lower() in answer.lower():
                        print(f"✅ Content check: PASSED")
                    else:
                        print(f"❌ Content check: FAILED (expected to contain '{test_case['expected_answer']}')")
                    
            else:
                print(f"❌ Status: HTTP {response.status_code}")
                print(f"Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ ERROR: Could not connect to the server")
            print("Make sure the server is running on localhost:8080")
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_public_zero_behavior()
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)
    print("\nKey Requirements:")
    print("✅ public=0 should ALWAYS return 'not enough information' when no relevant content")
    print("✅ public=1 should fallback to OpenAI general knowledge")
    print("✅ Default behavior (no public param) should be public=1")
