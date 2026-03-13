Model Count: 27 models ==> models covering all 24 backend endpoints the agent calls, all 4 specialist outputs, full conversation state, and the frontend API contract.


Category	                                    Models	                                                   Count
Enums	:-                          FlowType, ExperimentTypeHint, MessageType, UpdateIntent	                   4

Reference Data:-	         ZoneRef, CountryRef, BusinessFunctionRef, UseCaseRef, ExperimentTypeRef, KpiRef, ReferenceData	                                                                                                 7

User Context:-	                        UserProfile, ExperimentHistoryItem, UserContext	                       3

Specialist Outputs:	    HypothesisParserOutput, DateValidatorOutput, ColumnMetadata, ColumnDetection,           ColumnDetectorOutput, UpdateValidatorOutput	                                                                   6

Backend Payloads:-	      ExperimentCreatePayload, ExperimentUpdatePayload, SampleSizePayload,DataQualityPayload	                                                                                         4

Conversation:-	                    AgentMessage, CollectedFields, ConversationState	                       3

Agent API:-	                    AgentChatRequest, SuggestedOption, AgentChatResponse	                       3